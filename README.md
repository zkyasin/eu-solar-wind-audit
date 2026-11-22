# eu-solar-wind-audit

Checks how comparable published solar and wind production data are across Europe, against a
weather-based reference.

The project is in development.

## Real-time archive

RTE later replaces the éCO2mix real-time values with consolidated data. A scheduled workflow saves
the national real-time export from ODRÉ once a day and attaches it to the monthly release
`realtime-YYYY-MM`, with its retrieval time and SHA-256 checksum.
