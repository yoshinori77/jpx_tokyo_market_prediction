import pandas as pd
import time

from contextlib import contextmanager


# def get_time_info(df):
#     datetime_series = pd.to_datetime(df['Date']).dt
    
#     def get_weekday(dt_series):
#         return pd.get_dummies(dt_series.day_name())
    
#     def get_week(dt_series):
#         return dt_series.isocalendar().week
    
#     def get_month(dt_series):
#         return pd.get_dummies(dt_series.month_name())
    
#     weekday = get_weekday(datetime_series)
#     week = get_week(datetime_series)
#     month = get_month(datetime_series)
    
#     return pd.concat([df, weekday, week, month], axis=1)


@contextmanager
def timer(name: str):
    t0 = time.time()
    msg = f"[{name}] start"
    print(msg)
    yield

    msg = f"[{name}] done in {time.time() - t0:.2f} s"
    print(msg)
