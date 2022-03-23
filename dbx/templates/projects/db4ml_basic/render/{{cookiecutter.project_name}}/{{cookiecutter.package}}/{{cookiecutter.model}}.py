from typing import Dict

import pandas as pd
from db4ml import model, transform, pipeline, model_scoring
from mlflow.pyfunc import PyFuncModel
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import regexp_replace, substring, trim, round
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline as SKLPipeline


###
# Data Transformations
###
@transform(dataset="{{cookiecutter.model}}_initial")
def load(source_df: DataFrame, limit: int = -1) -> DataFrame:
    if limit > 0:
        source_df = source_df.limit(limit)

    return source_df.select(
        "loan_status",
        "int_rate",
        "revol_util",
        "issue_d",
        "earliest_cr_line",
        "emp_length",
        "verification_status",
        "total_pymnt",
        "loan_amnt",
        "grade",
        "annual_inc",
        "dti",
        "addr_state",
        "term",
        "home_ownership",
        "purpose",
        "application_type",
        "delinq_2yrs",
        "total_acc",
    )


@transform(dataset="{{cookiecutter.model}}_dataset")
def add_features({{cookiecutter.model}}_initial: DataFrame) -> DataFrame:
    df = {{cookiecutter.model}}_initial.filter(
        {{cookiecutter.model}}_initial.loan_status.isin(["Default", "Charged Off", "Fully Paid"])
    ).withColumn(
        "bad_loan", (~({{cookiecutter.model}}_initial.loan_status == "Fully Paid")).cast("float")
    )

    df = (
        df.withColumn("int_rate", regexp_replace("int_rate", "%", "").cast("float"))
            .withColumn("revol_util", regexp_replace("revol_util", "%", "").cast("float"))
            .withColumn("issue_year", substring(df.issue_d, 5, 4).cast("double"))
            .withColumn(
            "earliest_year", substring(df.earliest_cr_line, 5, 4).cast("double")
        )
    )
    df = df.withColumn("credit_length_in_years", (df.issue_year - df.earliest_year))

    df = df.withColumn(
        "emp_length", trim(regexp_replace(df.emp_length, "([ ]*+[a-zA-Z].*)|(n/a)", ""))
    )
    df = df.withColumn("emp_length", trim(regexp_replace(df.emp_length, "< 1", "0")))
    df = df.withColumn(
        "emp_length", trim(regexp_replace(df.emp_length, "10\\+", "10")).cast("float")
    )

    df = df.withColumn(
        "verification_status",
        trim(regexp_replace(df.verification_status, "Source Verified", "Verified")),
    )

    df = df.withColumn("net", round(df.total_pymnt - df.loan_amnt, 2))
    df = df.select(
        "bad_loan",
        "term",
        "home_ownership",
        "purpose",
        "addr_state",
        "verification_status",
        "application_type",
        "loan_amnt",
        "emp_length",
        "annual_inc",
        "dti",
        "delinq_2yrs",
        "revol_util",
        "total_acc",
        "credit_length_in_years",
        "int_rate",
        "net",
        "issue_year",
    )
    return df


###
# Model
###


class ConvertToCategorical(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):  # noqa
        return self

    def transform(self, X, y=None):  # noqa
        for col in X.columns:
            if X.dtypes[col] == "object":
                X[col] = X[col].astype("category").cat.codes
            X[col] = X[col].fillna(0)
        return X


@model(name="{{cookiecutter.model}}")
def create_model(params: Dict[str, str]):
    _pipeline = SKLPipeline(
        steps=[
            ("convert_to_cat", ConvertToCategorical()),
            ("cl", RandomForestClassifier(n_estimators=int(params["n_estimators"]))),
        ]
    )
    return _pipeline


###
# Batch Scoring
###
@pipeline(name="{{cookiecutter.model}}_scoring_spark")
@model_scoring(model_name="{{cookiecutter.model}}", spark_udf=True)
def {{cookiecutter.model}}_scoring_spark({{cookiecutter.model}}_dataset: DataFrame, spark: SparkSession, params: Dict[str, str]):
    {{cookiecutter.model}}_dataset.createOrReplaceTempView("loans")
    df = spark.sql(
        """
            select *, {{cookiecutter.model}}(term, home_ownership, purpose, addr_state, verification_status,
            application_type, loan_amnt, emp_length, annual_inc,dti, delinq_2yrs, revol_util, total_acc,
            credit_length_in_years, int_rate, net, issue_year) as prediction
            from loans
            """
    )
    df.show(10)
    df.write.format(params["format"]).mode("overwrite").saveAsTable(params["table_name"])


@pipeline(name="{{cookiecutter.model}}_scoring_pandas")
@model_scoring(model_name="{{cookiecutter.model}}", inject=True)
def {{cookiecutter.model}}_scoring_pandas(
    {{cookiecutter.model}}_dataset: pd.DataFrame, {{cookiecutter.model}}: PyFuncModel, spark: SparkSession, params: Dict[str, str]
):
    pdf = {{cookiecutter.model}}_dataset[
        [
            "term",
            "home_ownership",
            "purpose",
            "addr_state",
            "verification_status",
            "application_type",
            "loan_amnt",
            "emp_length",
            "annual_inc",
            "dti",
            "delinq_2yrs",
            "revol_util",
            "total_acc",
            "credit_length_in_years",
            "int_rate",
            "net",
            "issue_year",
        ]
    ]

    preds = {{cookiecutter.model}}.predict(pdf)
    pdf["predictions"] = preds
    sdf = spark.createDataFrame(pdf)
    sdf.show(10)
    sdf.write.format(params["format"]).mode("overwrite").saveAsTable(params["table_name"])
