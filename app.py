from pathlib import Path
from tempfile import TemporaryDirectory
from io import BytesIO

import geopandas as gpd
import pandas as pd
import streamlit as st
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union


# ── INNSTILLINGER ─────────────────────────────────────────────
st.set_page_config(page_title="Kraftfordeling", layout="wide")

DATAMAPPE = Path(__file__).resolve().parent


# ── HJELPEFUNKSJONER ──────────────────────────────────────────
def normaliser(tekst):
    return "".join(
        tegn for tegn in str(tekst).casefold()
        if tegn.isalnum()
    )


def finn_felt(data, navn):
    oppslag = {
        normaliser(kolonne): kolonne
        for kolonne in data.columns
    }

    for kandidat in navn:
        if normaliser(kandidat) in oppslag:
            return oppslag[normaliser(kandidat)]

    raise ValueError(
        f"Fant ikke feltet {navn[0]}. "
        f"Tilgjengelige felt: {list(data.columns)}"
    )


def polygoner(geometri):
    if geometri is None or geometri.is_empty:
        return Polygon()

    if isinstance(geometri, (Polygon, MultiPolygon)):
        return geometri

    deler = []

    for delgeometri in getattr(geometri, "geoms", []):
        polygon = polygoner(delgeometri)
        if not polygon.is_empty:
            deler.append(polygon)

    return unary_union(deler) if deler else Polygon()


def klargjor(data):
    if data.crs is None:
        raise ValueError("Laget mangler koordinatsystem.")

    if data.geometry.isna().any():
        raise ValueError(
            "Noen geometrier mangler eller kunne ikke repareres. "
            "Kontroller filen i QGIS."
        )

    data = data.to_crs(CRS).copy()
    data.geometry = data.geometry.make_valid().map(polygoner)

    return data[
        ~data.geometry.is_empty
        & (data.geometry.area > 0)
    ].copy()


def les(fil, lag=None):
    return gpd.read_file(
        fil,
        layer=lag,
        engine="pyogrio",
        on_invalid="fix",
    )


@st.cache_data
def les_bakgrunn():
    kommunefil = DATAMAPPE / "Kommuner.gml"
    kraftfil = DATAMAPPE / "NettKonsesjonsomrade.shp"

    for fil in [kommunefil, kraftfil]:
        if not fil.is_file():
            raise FileNotFoundError(f"Fant ikke {fil}")

    lag = gpd.list_layers(kommunefil)
    kommunelag = next(
        (
            navn for navn in lag["name"]
            if normaliser(navn) == "kommune"
        ),
        None,
    )

    if kommunelag is None:
        raise ValueError(
            "Fant ikke laget Kommune i Kommuner.gml."
        )

    kommuner = klargjor(les(kommunefil, kommunelag))

    nummerfelt = finn_felt(
        kommuner,
        [
            "KOMMUNENR-OFFISIELT",
            "kommunenummer",
            "kommunenr",
            "kommnr",
            "komm",
        ],
    )

    navnefelt = finn_felt(
        kommuner,
        [
            "KOMMUNENAVN-OFFISIELT",
            "kommunenavn",
            "kommnavn",
            "navn",
        ],
    )

    kommuner["Kommunenummer"] = (
        kommuner[nummerfelt]
        .astype("string")
        .str.strip()
        .str.replace(r"-\d+$", "", regex=True)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(4)
    )

    if not kommuner["Kommunenummer"].str.fullmatch(
        r"\d{4}", na=False
    ).all():
        raise ValueError("Kommunelaget har ugyldige kommunenummer.")

    kommuner["Kommune"] = (
        kommuner[navnefelt].astype("string").str.strip()
    )

    if (
        kommuner["Kommune"].isna()
        | kommuner["Kommune"].eq("")
    ).any():
        raise ValueError("Kommunelaget mangler kommunenavn.")

    kommuner = kommuner[
        ["Kommunenummer", "Kommune", "geometry"]
    ].dissolve(
        by="Kommunenummer",
        aggfunc="first",
        as_index=False,
    )

    kraft = klargjor(les(kraftfil))
    eierfelt = finn_felt(kraft, ["eierNavn"])

    kraft["Nettselskap"] = (
        kraft[eierfelt]
        .astype("string")
        .fillna("Ukjent nettselskap")
        .str.strip()
        .replace("", "Ukjent nettselskap")
    )

    kraft = kraft[
        ["Nettselskap", "geometry"]
    ].dissolve(
        by="Nettselskap",
        as_index=False,
    )

    return kommuner, kraft


def beregn(prosjekt, kommuner, kraft, minste_areal):
    prosjekt = klargjor(prosjekt)

    if prosjekt.empty:
        raise ValueError("Prosjektlaget inneholder ingen polygonarealer.")

    prosjektgeometri = prosjekt.geometry.union_all()
    xmin, ymin, xmax, ymax = prosjekt.total_bounds

    kommuner = kommuner.cx[
        xmin:xmax, ymin:ymax
    ].copy()

    kommuner.geometry = (
        kommuner.geometry.intersection(prosjektgeometri)
    )

    kommuner = klargjor(kommuner)

    if kommuner.empty:
        raise ValueError(
            "Prosjektet overlapper ingen kommuner. "
            "Kontroller koordinatsystemet."
        )

    uten_kommune = prosjektgeometri.difference(
        kommuner.geometry.union_all()
    ).area

    kommuner = kommuner[
        kommuner.geometry.area >= minste_areal * 1_000_000
    ].copy()

    kraft = kraft.cx[xmin:xmax, ymin:ymax].copy()

    kraftgeometri = (
        kraft.geometry.union_all()
        if not kraft.empty
        else Polygon()
    )

    rader = []
    advarsler = []

    if uten_kommune > 1:
        advarsler.append(
            f"{uten_kommune / 1_000_000:.6f} km² av prosjektet "
            "ligger utenfor kommunegrunnlaget."
        )

    for _, kommune in kommuner.iterrows():
        geometri = kommune.geometry
        prosjektareal = geometri.area
        sum_konsesjonsareal = 0

        def legg_til(navn, areal):
            if areal <= 0 or areal < minste_areal * 1_000_000:
                return

            rader.append({
                "Kommunenummer": kommune["Kommunenummer"],
                "Kommune": kommune["Kommune"],
                "Nettselskap": navn,
                "Prosjektareal km²": prosjektareal / 1_000_000,
                "Konsesjonsareal km²": areal / 1_000_000,
                "Andel %": 100 * areal / prosjektareal,
            })

        for _, selskap in kraft.iterrows():
            if geometri.intersects(selskap.geometry):
                areal = geometri.intersection(
                    selskap.geometry
                ).area

                sum_konsesjonsareal += areal
                legg_til(selskap["Nettselskap"], areal)

        udekket = geometri.difference(kraftgeometri).area
        legg_til("Uten konsesjonsdekning", udekket)

        if (
            sum_konsesjonsareal + udekket
            > prosjektareal * 1.0001
        ):
            advarsler.append(
                f"{kommune['Kommune']}: Konsesjonsområder "
                "fra ulike nettselskaper overlapper."
            )

    kolonner = [
        "Kommunenummer",
        "Kommune",
        "Nettselskap",
        "Prosjektareal km²",
        "Konsesjonsareal km²",
        "Andel %",
    ]

    tabell = pd.DataFrame(rader, columns=kolonner)

    if not tabell.empty:
        tabell = tabell.sort_values(
            ["Kommune", "Andel %"],
            ascending=[True, False],
        ).reset_index(drop=True)

        tabell = tabell.round({
            "Prosjektareal km²": 3,
            "Konsesjonsareal km²": 3,
            "Andel %": 2,
        })

    return tabell, advarsler


# ── NETTSIDEN ─────────────────────────────────────────────────
st.title("Kraftfordeling")
st.write(
    "Last opp prosjektområdet for å beregne areal "
    "og prosentvis fordeling per kommune og nettselskap."
)

opplastet = st.file_uploader(
    "Velg prosjektområde",
    type=["gml"],
)

if opplastet is not None:
    try:
        with TemporaryDirectory() as tempmappe:
            prosjektfil = Path(tempmappe) / "prosjekt.gml"
            prosjektfil.write_bytes(opplastet.getvalue())

            lag = gpd.list_layers(prosjektfil)

            polygonlag = lag[
                lag["geometry_type"].str.contains(
                    r"Polygon|Surface",
                    case=False,
                    na=False,
                )
            ]

            if polygonlag.empty:
                st.error("Filen inneholder ingen gjenkjente polygonlag.")
                st.stop()

            lagnavn = polygonlag["name"].tolist()

            aktuelle = [
                navn for navn in lagnavn
                if normaliser(navn) not in [
                    "kommune", "kommuner", "kommunegrense"
                ]
            ]

            standard = (
                lagnavn.index(aktuelle[0])
                if len(aktuelle) == 1
                else 0
            )

            valgt_lag = st.selectbox(
                "Prosjektlag",
                lagnavn,
                index=standard,
            )

            prosjekt = les(prosjektfil, valgt_lag)

            if prosjekt.crs is None:
                valgt_crs = st.selectbox(
                    "Filen mangler koordinatsystem. "
                    "Velg systemet koordinatene er lagret i:",
                    [
                        "Velg koordinatsystem",
                        "EPSG:25832",
                        "EPSG:25833",
                        "EPSG:25835",
                        "EPSG:4326",
                    ],
                )

                if valgt_crs == "Velg koordinatsystem":
                    st.info(
                        "Velg koordinatsystem for å fortsette."
                    )
                    st.stop()

                prosjekt = prosjekt.set_crs(valgt_crs)
            else:
                st.caption(
                    f"Koordinatsystem: {prosjekt.crs}"
                )

            minste_areal = st.number_input(
                "Minste areal som skal vises (km²)",
                min_value=0.0,
                value=5.0,
                step=1.0,
            )

            if st.button("Beregn fordeling", type="primary"):
                with st.spinner("Beregner …"):
                    kommuner, kraft = les_bakgrunn()

                    tabell, advarsler = beregn(
                        prosjekt,
                        kommuner,
                        kraft,
                        minste_areal,
                    )

                for melding in advarsler:
                    st.warning(melding)

                if tabell.empty:
                    st.info(
                        "Ingen rader oppfyller arealgrensen. "
                        "Prøv en lavere grense."
                    )
                else:
                    st.subheader("Resultat")

                    st.dataframe(
                        tabell,
                        hide_index=True,
                        use_container_width=True,
                    )

                    excel = BytesIO()

                    with pd.ExcelWriter(
                        excel, engine="openpyxl"
                    ) as writer:
                        tabell.to_excel(
                            writer,
                            sheet_name="Kraftfordeling",
                            index=False,
                        )

                    st.download_button(
                        "Last ned Excel",
                        data=excel.getvalue(),
                        file_name="kraftfordeling.xlsx",
                        mime=(
                            "application/vnd.openxmlformats-officedocument"
                            ".spreadsheetml.sheet"
                        ),
                        on_click="ignore",
                    )

    except Exception as feil:
        st.error(f"{type(feil).__name__}: {feil}")

st.caption(
    "100 % tilsvarer hele prosjektarealet i den enkelte kommunen. "
    "Arealgrensen skjuler små arealer uten å omberegne prosentene. "
    "Bakgrunnsdataene oppdateres ikke automatisk."
)
