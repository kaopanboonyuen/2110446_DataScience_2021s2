#
from airflow.utils.dates import days_ago
from airflow import DAG

from airflow.operators.bash import BashOperator

dag = DAG('cp_webcheck_failed', start_date=days_ago(1))

ping = BashOperator(task_id='http_check', bash_command='curl -s https://localhost > /dev/null', dag=dag)
inform = BashOperator(task_id='inform_status', bash_command='echo "CP website still works!"', dag=dag)

ping >> inform
