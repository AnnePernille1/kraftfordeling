# Installer: python -m pip install geopandas pyogrio shapely pandas streamlit openpyxl
# Start: python -m streamlit run app.py
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

# Bakgrunnsfilene ligger sammen med app.py.
DATAMAPPE = Path(__file__).resolve().parent

# Alle arealer beregnes i UTM 32.
CRS = "EPSG:25832"

# Brukes bare hvis koordinatsystem mangler i filen.
PROSJEKT_CRS = "EPSG:25833"
KOMMUNE_CRS = "EPSG:25833"
KRAFT_CRS = "EPSG:25832"


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


def klargjor(data, standard_crs=None):
    if data.crs is None:
        if standard_crs is None:
            raise ValueError("Laget mangler koordinatsystem.")

        data = data.set_crs(standard_crs)

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
        polygonlag = lag[
            lag["geometry_type"].str.contains(
                r"Polygon|Surface",
                case=False,
                na=False,
            )
        ]

        if len(polygonlag) == 1:
            kommunelag = polygonlag.iloc[0]["name"]
        else:
            raise ValueError(
                "Kunne ikke velge kommunelag automatisk.\n"
                + lag.to_string(index=False)
            )

    kommuner = klargjor(
        les(kommunefil, kommunelag),
        KOMMUNE_CRS,
    )

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
        raise ValueError(
            "Kommunelaget har ugyldige kommunenummer."
        )

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

    kraft = klargjor(
        les(kraftfil),
        KRAFT_CRS,
    )

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


def beregn_kraft(prosjekt, kommuner, kraft, minste_areal):
    prosjekt = klargjor(prosjekt, PROSJEKT_CRS)

    if prosjekt.empty:
        raise ValueError(
            "Prosjektlaget inneholder ingen polygonarealer."
        )

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
            "Kontroller at koordinatsystemene øverst i koden "
            "stemmer med filene."
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
            if areal <= 0:
                return

            if areal < minste_areal * 1_000_000:
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


    return tabell, advarsler


# Fylkesnavn kan korrigeres i nettsidens tabell ved behov.
FYLKER = {
    "03": "Oslo", "11": "Rogaland", "15": "Møre og Romsdal",
    "18": "Nordland", "31": "Østfold", "32": "Akershus",
    "33": "Buskerud", "34": "Innlandet", "39": "Vestfold",
    "40": "Telemark", "42": "Agder", "46": "Vestland",
    "50": "Trøndelag", "55": "Troms", "56": "Finnmark",
}


def arealtabell(prosjekt, kommuner, felt, fylketabell, minste_areal):
    prosjekt = klargjor(prosjekt, PROSJEKT_CRS)
    if prosjekt.empty:
        raise ValueError("Prosjektlaget inneholder ingen polygonarealer.")
    samlet = prosjekt.geometry.union_all()
    xmin, ymin, xmax, ymax = prosjekt.total_bounds
    deler = kommuner.cx[xmin:xmax, ymin:ymax].copy()
    deler.geometry = deler.geometry.intersection(samlet)
    deler = klargjor(deler)
    deler = deler[deler.geometry.area > 0].copy()
    if deler.empty:
        raise ValueError("Prosjektet overlapper ingen kommuner. Kontroller CRS.")
    deler = deler.sort_values("Kommune")
    resultat = deler[["Kommunenummer", "Kommune"]].reset_index(drop=True)
    resultat["Prosjektareal km²"] = deler.geometry.area.to_numpy() / 1_000_000

    # Union per kategori hindrer dobbelttelling av overlapp innen samme kategori.
    for kolonne in felt:
        verdier = prosjekt[kolonne].astype("string").str.strip().fillna("")
        verdier = verdier.replace("", "Ikke angitt")
        kategorier = sorted(verdier.unique())
        if normaliser(kolonne) == "fkbstandard":
            kategorier = sorted(set(kategorier) | {"A", "B", "C", "D"})
        for verdi in kategorier:
            navn = (f"FKB-{verdi} km²" if normaliser(kolonne) == "fkbstandard"
                    else f"{kolonne}: {verdi} km²")
            geometrier = prosjekt.loc[verdier.eq(verdi)].geometry
            geom = geometrier.union_all() if len(geometrier) else Polygon()
            resultat[navn] = deler.geometry.intersection(geom).area.to_numpy() / 1_000_000

    fylkeoppslag = fylketabell.set_index("Kommunenummer")["Fylkeskommune"]
    fylker = resultat["Kommunenummer"].map(fylkeoppslag).fillna("Ikke angitt")
    fylker = fylker.astype("string").str.strip().replace("", "Ikke angitt")
    for fylke in sorted(fylker.unique()):
        resultat[f"Fylke: {fylke} km²"] = resultat["Prosjektareal km²"].where(fylker.eq(fylke), 0.0)

    # Arealgrensen skjuler små celleverdier, uten å endre prosjektavgrensningen.
    arealfelt = [c for c in resultat if c.endswith(" km²")]
    for c in arealfelt:
        if c != "Prosjektareal km²":
            resultat[c] = resultat[c].where(resultat[c] >= minste_areal, 0.0)
    resultat = resultat[resultat["Prosjektareal km²"] >= minste_areal].copy()
    return resultat


def med_sum(tabell):
    if tabell.empty:
        return tabell
    summer = {c: tabell[c].sum() if c.endswith(" km²") else "" for c in tabell}
    summer["Kommune"] = "SUM"
    return pd.concat([tabell, pd.DataFrame([summer])], ignore_index=True)


# ── NETTSIDEN ─────────────────────────────────────────────────
st.title("Areal- og kraftfordeling")
st.write("Beregn areal per kommune, egenskapskategori, fylkeskommune og nettselskap.")
opplastet = st.file_uploader("Velg prosjektområde", type=["gml", "gpkg", "geojson"])
if opplastet is not None:
    try:
        with TemporaryDirectory() as tempmappe:
            prosjektfil = Path(tempmappe) / Path(opplastet.name).name
            prosjektfil.write_bytes(opplastet.getvalue())
            lag = gpd.list_layers(prosjektfil)
            polygonlag = lag[lag["geometry_type"].str.contains(r"Polygon|Surface", case=False, na=False)]
            if polygonlag.empty:
                st.error("Filen inneholder ingen gjenkjente polygonlag.")
                st.stop()
            lagnavn = polygonlag["name"].tolist()
            valgt_lag = st.selectbox("Prosjektlag", lagnavn)
            prosjekt = les(prosjektfil, valgt_lag)
            egenskaper = [c for c in prosjekt.columns if c != prosjekt.geometry.name
                          and normaliser(c) not in ["gmlid", "fid"]]
            standardfelt = [c for c in egenskaper if normaliser(c) in
                           ["fkbstandard", "informasjon", "punkttetthet", "laserpunkt", "laserunkttetthet"]]
            felt = st.multiselect("Egenskaper som skal fordeles i egne arealkolonner", egenskaper, default=standardfelt)
            st.caption("Velg for eksempel FKBSTANDARD, INFORMASJON eller feltet for laserpunkttetthet. Hver verdi får sin egen kolonne.")
            with st.expander("Se egenskapene i prosjektfilen"):
                st.dataframe(prosjekt[egenskaper], hide_index=True)
            kommuner, kraft = les_bakgrunn()
            prosjektklar = klargjor(prosjekt, PROSJEKT_CRS)
            if prosjektklar.empty:
                raise ValueError("Prosjektlaget inneholder ingen polygonarealer.")
            aktuelle = kommuner[kommuner.intersects(prosjektklar.geometry.union_all())]
            fylketabell = aktuelle[["Kommunenummer", "Kommune"]].sort_values("Kommune").reset_index(drop=True)
            fylketabell["Fylkeskommune"] = fylketabell["Kommunenummer"].str[:2].map(FYLKER).fillna("Ikke angitt")
            with st.expander("Kontroller fylkeskommune per kommune"):
                st.caption("Fylkestilhørighet fra kommunenummer. Kolonnene viser geografisk areal, ikke avtalt kostnadsandel.")
                fylketabell = st.data_editor(fylketabell, disabled=["Kommunenummer", "Kommune"], hide_index=True)
            minste_areal = st.number_input("Minste areal som skal vises (km²)", min_value=0.0, value=0.000005, step=0.000001, format="%.6f")
            if st.button("Beregn fordeling", type="primary"):
                with st.spinner("Beregner …"):
                    oversikt = arealtabell(prosjekt, kommuner, felt, fylketabell, minste_areal)
                    detalj, advarsler = beregn_kraft(prosjekt, kommuner, kraft, minste_areal)
                    if not detalj.empty:
                        nett = detalj.pivot_table(index="Kommunenummer", columns="Nettselskap", values="Konsesjonsareal km²", aggfunc="sum", fill_value=0)
                        nett.columns = [f"Nett: {c} km²" for c in nett.columns]
                        oversikt = oversikt.merge(nett, on="Kommunenummer", how="left")
                        for c in nett.columns:
                            oversikt[c] = oversikt[c].fillna(0)
                    oversikt = med_sum(oversikt)
                for melding in advarsler:
                    st.warning(melding)
                if oversikt.empty:
                    st.info("Ingen kommuner oppfyller arealgrensen.")
                else:
                    st.subheader("Arealfordeling per kommune (km²)")
                    st.dataframe(oversikt.style.format({c: "{:.3f}" for c in oversikt if c.endswith(" km²")}), hide_index=True, use_container_width=True)
                    st.caption("SUM gjelder de viste radene. Kategorier fra ulike felt kan dekke samme areal og skal ikke summeres på tvers av kolonner. 0,000 kan være et lite positivt areal; Excel inneholder full presisjon.")
                    with st.expander("Kraftfordeling med prosentandeler"):
                        st.dataframe(detalj, hide_index=True)
                    excel = BytesIO()
                    with pd.ExcelWriter(excel, engine="openpyxl") as writer:
                        oversikt.to_excel(writer, sheet_name="Arealfordeling", index=False)
                        detalj.to_excel(writer, sheet_name="Kraftfordeling", index=False)
                        for ark in writer.sheets.values():
                            ark.freeze_panes = "C2"
                            ark.auto_filter.ref = ark.dimensions
                            for kol in ark.columns:
                                ark.column_dimensions[kol[0].column_letter].width = min(45, max(14, len(str(kol[0].value)) + 2))
                                for celle in kol[1:]:
                                    if isinstance(celle.value, (float, int)):
                                        celle.number_format = "0.000000"
                    st.download_button("Last ned Excel", data=excel.getvalue(), file_name="arealfordeling.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", on_click="ignore")
    except Exception as feil:
        st.error(f"{type(feil).__name__}: {feil}")
st.caption("Arealer beregnes fra geometrien i UTM 32. Bakgrunnsfiler oppdateres ikke automatisk.")
