FROM pgvector/pgvector:pg17

ENV POSTGRES_DB=bjj_rules

# Scripts in this directory run once, when the data directory is first initialized.
COPY db/create_rules_table.sql /docker-entrypoint-initdb.d/

EXPOSE 5432
