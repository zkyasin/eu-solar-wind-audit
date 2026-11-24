# Data licences

This matrix backs the [publication rules](evaluation-plan.md#publication-rules) of the evaluation
plan. A source is used on the public page only as far as its row allows. A row marked *to verify*
is treated as *Nothing* until its terms are read and the row is updated.

What the project may publish:

- **Raw values**: the series as published, at its native or hourly resolution.
- **Derived only**: monthly aggregates, flags and classes computed from the data; never the hourly
  values themselves.
- **Metrics only**: error or comparison statistics against this project's own outputs.
- **Nothing**: used internally, or not used, until the terms are confirmed.

## Matrix

| Source | Provider | Dataset(s) used | Licence | Required attribution | May publish | Status |
|---|---|---|---|---|---|---|
| ODRÉ éCO2mix | RTE | éCO2mix national and regional, consolidated, definitive and real time; national register of production installations at the end of 2024 | [Licence Ouverte 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence) | Producer name and date of last update | Raw values | Verified 2026-11-22, [dataset metadata](https://odre.opendatasoft.com/explore/dataset/eco2mix-national-cons-def/information/) |
| RTE Analyses et données | RTE | Monthly negative-price modulation volumes (CSV) | Site terms: reuse permitted with the source mentioned | Source: RTE | Raw values (monthly) | Verified 2026-11-22, [mentions légales](https://analysesetdonnees.rte-france.com/mentions-legales) section 5 |
| SMARD | Bundesnetzagentur | Generation, consumption | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `Bundesnetzagentur \| SMARD.de` | Raw values | Verified 2026-11-22, [Datennutzung](https://www.smard.de/en/datennutzung) |
| SMARD prices | Bundesnetzagentur | Day-ahead wholesale prices | Under review (HEDWIG proceedings) | `Bundesnetzagentur \| SMARD.de` | Derived only (price classes) | Verified 2026-11-22, [Datennutzung](https://www.smard.de/en/datennutzung) |
| ENTSO-E Transparency Platform | ENTSO-E (primary owners: TSOs, NEMOs) | 16.1.b/c actual generation, 14.1.a installed capacity, 14.1.d wind and solar forecasts, 6.1.a actual load, 12.1.d day-ahead prices | [Terms of Use](https://transparencyplatform.zendesk.com/hc/en-us/articles/40921911218961-Legal-Terms-and-Conditions) (2023); these items are not on the CC BY 4.0 list | Source of publication: ENTSO-E Transparency Platform | Derived only, not relicensed under CC BY | Verified 2026-11-22, Terms of Use clause 3.1 and List of data available for free re-use (18 October 2023) |
| ERA5 (CDS) | Copernicus Climate Change Service (C3S), ECMWF | ERA5 hourly single levels, ERA5T | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Copernicus line (see Attribution) and citation, DOI 10.24381/cds.adbb2d47 | Raw values | Verified 2026-11-22, [CDS catalogue](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview) |
| Icechunk ERA5 | Earthmover (AWS Open Data) | ERA5 single levels, Icechunk store | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Earthmover and the registry link, plus the Copernicus line and ERA5 citation | Raw values | Verified 2026-11-22, [registry entry](https://registry.opendata.aws/earthmover-era5/) |
| ARCO-ERA5 | Google Research | ERA5 Zarr stores on Google Cloud | ERA5 terms; the README links the older Licence to use Copernicus Products | Copernicus line | Raw values | Verified 2026-11-22, [README](https://github.com/google-research/arco-era5#can-i-use-the-data-for-researchcommercial-purposes) |
| Eurostat | European Commission (Eurostat) | Annual gross electricity production by fuel (nrg_bal_peh) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `Source: <dataset DOI or link>, <access date>`; state modifications with a non-responsibility disclaimer | Raw values | Verified 2026-11-22, [copyright notice](https://ec.europa.eu/eurostat/help/copyright-notice) |
| SDES | Ministry for the Ecological Transition (SDES) | Quarterly solar and wind capacity tables; yearly self-consumption by technology (data file of « L'autoconsommation d'électricité en France », 2 October 2026) | Open licence under décret n° 2017-638 | `Service des données et études statistiques (SDES), <publication>` and date of last update; data not altered | Raw values | Verified 2026-11-22, [mentions légales](https://www.statistiques.developpement-durable.gouv.fr/mentions-legales) |
| Enedis open data | Enedis | Production capacity by size class, half-hourly production aggregates | [Licence Ouverte 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence), per dataset | Producer name and date of last update | Raw values | Organisation listing verified 2026-11-22, [data.gouv.fr organisation](https://www.data.gouv.fr/organizations/electricite-reseau-distribution-france/); each dataset to verify |
| Elia Open Data | Elia Transmission Belgium | Solar and wind measured and upscaled production | [Elia Open Data Licence](https://opendata.elia.be/pages/licence/) (CC BY 4.0 under Belgian law) | `Elia Transmission Belgium SA` | Raw values | Verified 2026-11-22, [licence page](https://opendata.elia.be/pages/licence/) |
| Energi Data Service | Energinet | Production per price area (DK1, DK2) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `Source: Energinet (www.energidataservice.dk).` | Raw values | Verified 2026-11-22, [terms and conditions](https://www.energidataservice.dk/terms-and-conditions) |
| REData | Red Eléctrica (REE) | Generation by technology | REE website legal notice: information and non-commercial use only | Source: Red Eléctrica, date of last update; meaning not altered | Derived only | Verified 2026-11-22, [legal notice](https://www.ree.es/en/legal-notice-privacy-policy) section 5.1 |
| ESIOS | Red Eléctrica (REE) | Curtailment and technical restriction volumes | Unknown | Unknown | Nothing | To verify |
| Energy-Charts | Fraunhofer ISE | Generation series, installed capacity | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) unless a response states otherwise | `Energy-Charts.info` | Raw values | Verified 2026-11-22, [API documentation](https://api.energy-charts.info/) |
| Energy-Charts prices | Fraunhofer ISE | Day-ahead prices | CC BY 4.0 from SMARD for 16 zones; private and internal use only for the others, raw or derived | `Bundesnetzagentur \| SMARD.de` for the CC BY zones | CC BY zones: as SMARD prices; others: Nothing | Verified 2026-11-22, [API documentation](https://api.energy-charts.info/), `/price` |
| Renewables.ninja | Renewables.ninja (Pfenninger, Staffell) | National PV and wind capacity factors | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) | Cite the two 2016 papers or link to www.renewables.ninja | Metrics only | Verified 2026-11-22, [about page](https://www.renewables.ninja/about) |
| PECD 4.2 | C3S with ENTSO-E | Solar and wind capacity factors, reanalysis | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `Copyright © <year> ECMWF/C3S/ENTSO-E`, disclaimer and catalogue citation, DOI 10.24381/cds.f323c5ec | Metrics only | Verified 2026-11-22, [CDS catalogue](https://cds.climate.copernicus.eu/datasets/sis-energy-pecd?tab=overview) |
| C3S energy indicators | C3S | Global climate and energy indicators derived from reanalysis (successor of the Europe dataset, which stopped updating on 2026-07-01) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Copernicus line and citation, DOI 10.24381/3bb607bd | Metrics only | Verified 2026-11-21, [CDS catalogue](https://cds.climate.copernicus.eu/datasets/sis-energy-global-reanalysis?tab=overview) |
| SARAH-3 | EUMETSAT CM SAF | Surface irradiance | CC BY 4.0 by EUMETSAT ([CM SAF rights page](https://wui.cmsaf.eu/safira/action/viewImpressum)) | `Copyright <year> EUMETSAT` and CM SAF acknowledgement | Metrics only | Verified 2026-11-22; two official texts differ (note 17) |
| GWA bias factors | Karkossa et al. (Zenodo) | ERA5 bias-correction factors (GWA v2, SARAH-3) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Dataset citation, DOI 10.5281/zenodo.23686879 | Raw values | Verified 2026-11-22, [Zenodo record](https://zenodo.org/records/23686878) |
| MaStR | Bundesnetzagentur | Unit register, quarterly public export of 2026-10-01 (solar and wind unit tables, grid connection points) | [dl-de/by-2-0](https://www.govdata.de/dl-de/by-2-0) | `Bundesnetzagentur, Marktstammdatenregister`, licence notice with link, dataset URI; mark changed data | Raw values | Verified 2026-11-22, [Datendownload](https://www.marktstammdatenregister.de/MaStR/Datendownload) and [Impressum](https://www.marktstammdatenregister.de/MaStR/Startseite/Impressum) |
| NREL turbine-models | NREL, Alliance for Sustainable Energy (repository now under NatLabRockies) | Power curves of twelve turbines, copied into `src/eu_solar_wind_audit/data/` | [BSD 3-Clause](https://github.com/NatLabRockies/turbine-models/blob/main/LICENSE.txt) | Copyright notice and licence text kept with the copied tables | Raw values | Verified 2026-11-22, [LICENSE.txt](https://github.com/NatLabRockies/turbine-models/blob/main/LICENSE.txt) |
| Natural Earth | Natural Earth | Admin-1 states and provinces, 1:10m, release v5.1.2 | Public domain ([terms of use](https://www.naturalearthdata.com/about/terms-of-use/)) | None required; optional `Made with Natural Earth.` | Raw values | Verified 2026-11-22, [terms of use](https://www.naturalearthdata.com/about/terms-of-use/) |
| BKG VG5000 | Bundesamt für Kartographie und Geodäsie (BKG) | Verwaltungsgebiete 1:5 000 000, state 2026-01-01: districts and Länder | [dl-de/by-2-0](https://www.govdata.de/dl-de/by-2-0) | `© BKG (<year>) dl-de/by-2-0 (Daten verändert), Datenquellen: https://sgx.geodatenzentrum.de/web_public/gdz/datenquellen/datenquellen_vg_nuts.pdf` | Raw values | Verified 2026-11-22, [product page](https://gdz.bkg.bund.de/index.php/default/verwaltungsgebiete-1-5-000-000-stand-01-01-vg5000-01-01.html), [source note guidance](https://gdz.bkg.bund.de/index.php/default/hinweise-zu-nutzungsbedingungen-und-quellenvermerken/) and `nutzungsbedingungen_vg5000.pdf` in the download |
| EMODnet Human Activities | CETMAR for EMODnet | Wind farms, polygon layer (`EMODnet_HA_Energy_WindFarms_20260710`) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | `This data was downloaded from the EMODnet Portal (https://emodnet.ec.europa.eu/en/). The data originator is CETMAR.` | Raw values | Verified 2026-11-22, [metadata record](https://ows.emodnet-humanactivities.eu/geonetwork/srv/api/records/8201070b-4b0b-4d54-8910-abcea5dce57f) and [terms of use](https://emodnet.ec.europa.eu/en/terms-use-emodnet-online-services-data-and-data-products) |

## Notes

1. **ODRÉ éCO2mix.** The four éCO2mix datasets on ODRÉ (national and regional, consolidated and
   definitive, real time) carry Licence Ouverte v2.0 with RTE as publisher. The licence asks for
   the producer name and the date of last update, and forbids suggesting endorsement.
2. **RTE Analyses et données.** No named licence: the French legal notice states that the data
   published on the site are public and that reuse is permitted if the source is mentioned. The
   English page names `www.rte-france.com` instead; the French text is used.
3. **SMARD prices.** The Bundesnetzagentur is examining the licensing of wholesale prices as part
   of the HEDWIG proceedings. Until it concludes, prices from SMARD, and from any mirror of SMARD,
   appear only as the price classes of the evaluation plan.
4. **ENTSO-E.** The [List of data available for free re-use](https://transparencyplatform.zendesk.com/hc/article_attachments/40921869379729)
   (18 October 2023) is under CC BY 4.0, but it covers only load forecasts, grid, cross-zonal
   capacity, flow, redispatch and balancing items. Actual generation (16.1), installed capacity and generation
   forecasts (14.1), actual load (6.1.a) and day-ahead prices (12.1.d) are not on it. For those,
   clause 3.1 of the Terms of Use applies: mention the platform as the source, no implied
   endorsement, and no prejudice to the primary owner's rights. Values derived from these items
   are therefore published only as monthly aggregates, flags or classes, and are not relicensed
   under CC BY without written confirmation.
5. **ERA5.** The CDS catalogue now lists ERA5 under CC BY 4.0. Model outputs computed from ERA5
   are modified information and use the "Contains modified" form of the Copernicus line.
6. **Icechunk ERA5.** The open store holds final ERA5 (to 2026-06-30 when read on 2026-11-21) and
   is updated quarterly. Recent days (ERA5T) come from ARCO-ERA5, also anonymous; the attribution
   is the same.
7. **ARCO-ERA5.** The README still points to the Licence to use Copernicus Products, which the CDS
   has replaced with CC BY 4.0 for ERA5. Both require the Copernicus attribution; the CDS
   licence is the one applied.
8. **Eurostat.** Statistical data may be reused for any purpose with the source acknowledged.
   Changes must be stated, with a disclaimer of Eurostat's responsibility; this applies to the
   harmonized annual totals.
9. **SDES.** Free reuse, including commercial, provided the data are not altered and the source
   and date of last update are cited. Capacities converted to AC are labelled as derived, next to
   the SDES value.
10. **Enedis.** 98 of the first 100 Enedis datasets listed on data.gouv.fr are under Licence
    Ouverte 2.0; one is under ODbL. The licence is checked for each dataset before it is added.
11. **Elia.** The licence page states that the data are governed by CC BY 4.0 under Belgian law.
    The attribution name comes from the dataset metadata (`attributions` field).
12. **REData.** The REE legal notice allows personal copies and non-commercial content for
    information purposes, with the source, the date of last update and the meaning preserved. It
    does not name an open licence, so Spanish values stay at the derived level until REE confirms
    otherwise.
13. **ESIOS.** The legal notice is rendered client-side and could not be read. The REE curtailment
    volumes named in the evaluation plan stay unpublished until the terms are confirmed.
14. **Energy-Charts prices.** The CC BY 4.0 zones are AT, BE, CH, CZ, DE-LU, DE-AT-LU, DK1, DK2,
    FR, HU, IT-North, NL, NO2, PL, SE4 and SI, all from SMARD (so note 3 applies). For every other
    zone, including ES, PT and the Italian zones other than IT-North, any external use is
    prohibited, raw or derived. Energy-Charts is not used as a price source for these zones.
15. **Renewables.ninja.** CC BY-NC 4.0 allows non-commercial sharing only. Publishing only
    comparison metrics keeps the non-commercial condition off the project's published values.
16. **PECD and C3S energy indicators.** The licence would allow raw values; they are used as
    baselines, so only metrics are published. The PECD catalogue entry contains v4.1 (deprecated)
    and v4.2; only v4.2 is used. Its citation and attribution tab differs from the other C3S
    entries: a copyright line for ECMWF, C3S and ENTSO-E and an ECMWF, C3S and ENTSO-E
    disclaimer instead of the Copernicus line.
17. **SARAH-3.** Two official texts differ: the CM SAF rights page states CC BY 4.0 by EUMETSAT,
    while the [CM SAF products licence on the Copernicus catalogues](https://ewds.climate.copernicus.eu/licences/eumetsat-cm-saf)
    grants free use with the line `copyright (<year>) EUMETSAT` and a CM SAF acknowledgement. Both
    lines are given until the SARAH-3 product page is checked.
18. **GWA bias factors.** Zenodo record 23686878 is the concept record; the current version is
    23686879 (published 2026-10-27). The factors are applied to ERA5, and only corrected model
    outputs are published.
19. **MaStR.** dl-de/by-2-0 requires the provider name as the provider gives it, the licence notice
    with its link, the dataset URI, and a note when the data were changed. The download page states
    the licence for each export (the Impressum states it only for the .xls and .csv files and the
    web service), and neither page gives a wording for the provider name, so the line names the
    Bundesnetzagentur and the register. The export documentation (revision 26.1.2) states that the
    daily export gets a new file name each day to prevent automated retrieval (section 2.3) and
    keeps quarterly exports for analyses (section 3.3), so the project uses the quarterly export of
    2026-10-01: its URL is the dataset URI, and it is downloaded once, checked against its SHA-256
    and cached. The site terms (Impressum) add that retrieval must not impair the use of the site
    by other visitors. District capacities are aggregates computed from the unit data, so they are
    marked as changed data.
20. **NREL turbine-models.** The BSD 3-Clause licence (Copyright (c) 2020, Alliance for Sustainable
    Energy, LLC) covers the whole repository, data folders included. The speed and power columns of
    twelve tables were copied unchanged from commit `e9e4ccc` (2026-04-03); the licence text ships
    next to them as `turbine-models-LICENSE.txt`. Two other sources were rejected: the power curves
    of the Virtual Wind Farm model are excluded from its BSD licence and "remain the property of the
    respective copyright holders", and the OpenEnergy turbine library is under ODbL 1.0
    (share-alike) with manufacturer sources that carry no licence.
21. **Natural Earth.** All Natural Earth vector data are in the public domain and crediting the
    authors is optional. The terms page also lists data the authors received under non-exclusive
    licences to build the base map (The Washington Post base map data, JRC rivers and lakes of
    Europe, XNR roads of North America, IMA time zones) and Wikidata names under CC0, and still
    places all Natural Earth data in the public domain. The admin-1 polygons serve only to spread
    capacity over the weather grid.
22. **BKG VG5000.** The terms in the download (dated 2025-06-30) and the product page require
    the source line on every public display, with `<year>` the year of the last download, and a
    change notice. The BKG guidance on source notes gives the notice as `(Daten verändert)` after
    the licence name. The boundaries are changed: reprojected to WGS 84, with the coastal sea and
    the Bodensee left out. On a web page, "BKG" links to https://www.bkg.bund.de and
    "dl-de/by-2-0" to the licence.
23. **EMODnet wind farms.** The metadata record grants CC BY 4.0 and gives the attribution line,
    naming CETMAR as the data originator. The EMODnet terms of use ask users to cite the originator
    named in the metadata and to check its use restrictions; the record lists the national
    authorities, OSPAR, HELCOM, WindEurope, Global Energy Monitor and The Wind Power as sources
    of the aggregated layer and states no other restriction. The vintage (revision date
    2026-07-10) is part of the dataset identifier.

## Attribution

The public page footer lists the lines of the sources shown on that page. Placeholders in angle
brackets are filled at build time from the data vintage.

```text
France: RTE, éCO2mix data via ODRÉ, Licence Ouverte 2.0, last updated <date>.
France, negative-price modulation: source RTE, analysesetdonnees.rte-france.com, last updated <date>.
Germany: Bundesnetzagentur | SMARD.de, CC BY 4.0.
Germany, unit register: Bundesnetzagentur, Marktstammdatenregister, dl-de/by-2-0 (www.govdata.de/dl-de/by-2-0), <dataset URI>; data changed.
Source of publication: ENTSO-E Transparency Platform (transparency.entsoe.eu). Values shown are derived from the published data.
Belgium: Elia Transmission Belgium SA, Elia Open Data, CC BY 4.0.
Source: Energinet (www.energidataservice.dk). CC BY 4.0.
Spain: source Red Eléctrica (REData), last updated <date>.
Energy-Charts.info, Fraunhofer ISE, CC BY 4.0.
Source: Eurostat, <dataset DOI or link>, <access date>. Data modified by this project; Eurostat is not responsible for the modified data.
Service des données et études statistiques (SDES), <publication>, last updated <date>.
Enedis open data, <dataset>, Licence Ouverte 2.0, last updated <date>.
Contains modified Copernicus Climate Change Service information <year>. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains.
Copernicus Climate Change Service (2023): ERA5 hourly data on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI: 10.24381/cds.adbb2d47 (Accessed on <date>).
ERA5 accessed through Icechunk ERA5 (Earthmover), https://registry.opendata.aws/earthmover-era5, CC BY 4.0, accessed <date>.
Copyright <year> EUMETSAT. This work used data from EUMETSAT's Satellite Application Facility on Climate Monitoring (CM SAF), SARAH-3.
Turbine power curves: NREL turbine-models, Copyright (c) 2020 Alliance for Sustainable Energy, LLC, BSD 3-Clause.
ERA5 bias-correction factors: Karkossa, Grochowicz, Koivisto, Hahmann and Victoria (2026), Zenodo, doi:10.5281/zenodo.23686879, CC BY 4.0.
Made with Natural Earth.
Germany, administrative boundaries: © BKG (<year>) dl-de/by-2-0 (Daten verändert), Datenquellen: https://sgx.geodatenzentrum.de/web_public/gdz/datenquellen/datenquellen_vg_nuts.pdf. Changes: reprojected to WGS 84; coastal sea and Bodensee removed.
Offshore wind farms: this data was downloaded from the EMODnet Portal (https://emodnet.ec.europa.eu/en/). The data originator is CETMAR. <dataset identifier>, CC BY 4.0.
Baselines: Renewables.ninja (Pfenninger and Staffell 2016; Staffell and Pfenninger 2016), CC BY-NC 4.0.
Copernicus Climate Change Service (2024): Climate and energy related variables from the Pan-European Climate Database derived from reanalysis and climate projections. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI: 10.24381/cds.f323c5ec (Accessed on <date>). PECD v4.2, modified.
Copyright © <year> ECMWF/C3S/ENTSO-E. ECMWF & C3S & ENTSO-E does not accept any liability whatsoever for any error or omission in the data, their availability, or for any loss or damage arising from their use.
Copernicus Climate Change Service (2026): Global climate and energy indicators from 1950 to present derived from reanalysis. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI: 10.24381/3bb607bd (Accessed on <date>).
```
