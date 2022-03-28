from airflow import DAG
from airflow.operators.python import BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime


def _choose_best_model(accuracy):
    if accuracy > 0.8:
        return 'accurate'
    return 'inaccurate'


with DAG('branch_oper', start_date=datetime(2021, 1, 1), catchup=False) as dag:
    choose_best_model = BranchPythonOperator(task_id='choose_best_model', python_callable=_choose_best_model, op_args=[0.75])
    accurate = DummyOperator(task_id='accurate')
    inaccurate = DummyOperator(task_id='inaccurate')

    choose_best_model >> [accurate, inaccurate]
