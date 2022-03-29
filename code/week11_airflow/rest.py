#
import json

from airflow.utils.dates import days_ago
from airflow import DAG

from airflow.providers.http.operators.http import SimpleHttpOperator

with DAG('rest_test', start_date=days_ago(1)) as dag:
    post_op = SimpleHttpOperator(
        task_id='post_op',
        endpoint='items',
        http_conn_id='localhttp',
        data=json.dumps({"name": 'test', 'price': 0.5, 'tax': 1}),
        headers={"Content-Type": "application/json"},
        response_check=lambda response: response.json()['price_with_tax'] == 1.5
    )

    post_op
