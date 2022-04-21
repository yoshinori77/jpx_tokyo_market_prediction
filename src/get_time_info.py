def get_time_info(df):
    datetime_series = pd.to_datetime(df['Date']).dt
    
    def get_weekday(dt_series):
        return pd.get_dummies(dt_series.day_name())
    
    def get_week(dt_series):
        return dt_series.isocalendar().week
    
    def get_month(dt_series):
        return pd.get_dummies(dt_series.month_name())
    
    weekday = get_weekday(datetime_series)
    week = get_week(datetime_series)
    month = get_month(datetime_series)
    
    return pd.concat([df, weekday, week, month], axis=1)