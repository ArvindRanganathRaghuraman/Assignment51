CREATE OR REPLACE STAGE nvidia_stage

LIST nvidia_stage;

REMOVE @nvidia_stage/NVDA_Historical_Statistics_2020_2025_fixed.csv.gz;


CREATE OR REPLACE TABLE NVIDIA_FINANCIALS (
    asOfDate TIMESTAMP,
    EnterpriseValue FLOAT,
    EnterprisesValueEBITDARatio FLOAT,
    EnterprisesValueRevenueRatio FLOAT,
    ForwardPeRatio FLOAT,
    MarketCap FLOAT,
    PbRatio FLOAT,
    PeRatio FLOAT,
    PegRatio FLOAT,   -- Newly added
    PsRatio FLOAT     -- Newly added
);


COPY INTO NVIDIA_FINANCIALS
FROM @nvidia_stage/nvidia_pivoted_cleaned_data.csv
FILE_FORMAT = (TYPE = CSV, SKIP_HEADER = 1);

SELECT * FROM NVIDIA_FINANCIALS LIMIT 10;