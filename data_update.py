from scripts.util.data.DataUpdater import update_main
from scripts.util.data.sqlConnector import update_sql_since , update_sql_dates


update_main()
update_sql_since()
# update_sql_dates(20990101 , 20241231)
