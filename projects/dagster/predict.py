import decimal

from dagster import job, op
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.types import StructType, StructField, StringType, DateType, DecimalType
from pyspark.sql import functions
import pandas as pd
from pmdarima.arima import auto_arima

@op(required_resource_keys={"pyspark"})
def predict_op(context, dependent_job=None):
    spark = context.resources.pyspark.spark_session
    sybmbols_df = spark.sql("select distinct symbol from warehouse.silver.stock_markets_with_relative_prices")
    symbols = sybmbols_df.rdd.map(lambda row: row[0]).collect()
    result_data = []
    n_periods=7

    for symbol in symbols:
        # price_close --> price_close_relative
        df = spark.sql(f"select dt, price_close from warehouse.silver.stock_markets_with_relative_prices where symbol in ('{symbol}') and dt between '2020-01-01' and '2022-06-30' order by dt")
        data = df.toPandas()
        train_data = data[:len(data) - n_periods]
        # context.log.info(f"train_data: {train_data}")
        test_data = data[-n_periods:]
        # context.log.info(f"test_data: {test_data}")
        # price_close --> price_close_relative
        arima_model = auto_arima(train_data["price_close"], start_p=1, start_d=1, start_q=0, max_p=5, max_d=5, max_q=5, start_P=0, start_D=1, start_Q=0, max_P=5, max_D=5, max_Q=5, m=11, seasonal=True, random_state=20, supress_warning=True, stepwise=True)
        # arima_model.summary()

        # test_data.set_index('dt', inplace=True)

        #  RangeIndex(start=600, stop=629, step=1)
        # {type(test_data.iloc[0]["dt"])}  
        # context.log.info(f"index: {test_data.index} symbol: {symbol} ")
        predicted_data = pd.DataFrame(arima_model.predict(n_periods=n_periods), index=test_data.index)
        # predicted_data = pd.DataFrame(arima_model.predict(n_periods=n_periods), index=test_data['dt'][:])
        
        # context.log.info(f"result_data: {predicted_data}")
        
        i=0
        for row in predicted_data.itertuples():
            d = test_data.iloc[i]["dt"]
            # context.log.info(f"row {d}")
            # row[0]
            result_data.append([symbol, d, decimal.Decimal(row[1])])
            i+=1

    # context.log.info(f"result_data: {result_data}")

    schema = StructType([
        StructField('symbol', StringType(), True),
        StructField('dt', DateType(), True),
        StructField('price_predicted', DecimalType(32,16), True),
    ])
    # context.log.info(f"schema: {schema}")
    r = spark.createDataFrame(result_data, schema)
    # context.log.info(f"createDataFrame: {r}")
    r.writeTo("warehouse.silver.predicted_data").using("iceberg").tableProperty("write.format.default", "parquet").createOrReplace()
    

