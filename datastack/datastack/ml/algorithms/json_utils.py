import json
import pandas as pd


class JsonConverter():

    @staticmethod
    def split_array(s, title="PC"):
        l = json.loads(s)
        t = [title+str(i+1) for i in range(len(l))]
        return dict(zip(t, l))

    @staticmethod
    def split_series(series, title="PC"):
        result = series.apply(lambda s: pd.Series(JsonConverter.split_array(s=s, title=title)))
        t = [title+str(i+1) for i in range(len(result.columns))]
        return result[t]

    @staticmethod
    def split_column_df(df, column, title=None):
        if title is None:
            title = column
        col = JsonConverter.split_series(series=df[column], title=title)
        for column in col.columns:
            df[column] = col[column]
        return df, list(col.columns)