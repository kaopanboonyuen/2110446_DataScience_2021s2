#
from airflow.utils.dates import days_ago
from airflow import DAG

from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator

# 'with' enables DAG to become context managers; automatically assign new operators to that DAG
with DAG('concurrent_dag', start_date=days_ago(1)) as dag:
    start = DummyOperator(task_id='start_task')
    ping = BashOperator(task_id='cp_check', bash_command='curl https://www.cp.eng.chula.ac.th')
    ping2 = BashOperator(task_id='eng_check', bash_command='curl https://www.eng.chula.ac.th')
    inform = BashOperator(task_id='inform_status', bash_command='echo "CP website still works!"')

    # creating DAG dependencies
    start >> [ping, ping2] >> inform
