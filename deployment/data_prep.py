
import pandas as pd
import numpy as np

def load_data_from_csv():
    try:
        users_df = pd.read_csv('/datasets/ds_s13_users.csv')
        visits_df = pd.read_csv('/datasets/ds_s13_visits.csv')
        ads_activity_df = pd.read_csv('/datasets/ads_activity.csv')
        surf_depth_df = pd.read_csv('/datasets/surf_depth.csv')
        primary_device_df = pd.read_csv('/datasets/primary_device.csv')
        cloud_usage_df = pd.read_csv('/datasets/cloud_usage.csv')
    except:
        users_df = pd.read_csv('https://code.s3.yandex.net/datasets/ds_s13_users.csv')
        visits_df = pd.read_csv('https://code.s3.yandex.net/datasets/ds_s13_visits.csv')
        ads_activity_df = pd.read_csv('https://code.s3.yandex.net/datasets/ads_activity.csv')
        surf_depth_df = pd.read_csv('https://code.s3.yandex.net/datasets/surf_depth.csv')
        primary_device_df = pd.read_csv('https://code.s3.yandex.net/datasets/primary_device.csv')
        cloud_usage_df = pd.read_csv('https://code.s3.yandex.net/datasets/cloud_usage.csv')    

    df_dict = {
        'users_df': users_df,
        'visits_df': visits_df,
        'ads_activity_df': ads_activity_df,
        'surf_depth_df': surf_depth_df,
        'primary_device_df': primary_device_df,
        'cloud_usage_df': cloud_usage_df}

    return df_dict

def merge_tables(df_dict):

    '''Нормализуем данные в категориальных столбцах'''
    for df_name, df in df_dict.items():
        for col in df.select_dtypes(include=['object']).columns:
            df_dict[df_name][col] = df_dict[df_name][col].str.lower()
            df_dict[df_name][col] = df_dict[df_name][col].str.strip()

    '''Чистим от дубликатов'''
    for df_name, df in df_dict.items():
        df_dict[df_name] = df.drop_duplicates()

    '''Создаем агрегированные признаки количества сессий по категориям для каждого пользователя из visits_df'''
    session_per_category = pd.pivot_table(df_dict['visits_df'],
                                          index='user_id',
                                          columns='website_category',
                                          values='session_id',
                                          aggfunc='nunique',
                                          fill_value=0)

    # Убираем отображение имени колонок в pivot
    session_per_category.columns.name = None
    # Расчет количества сессий для каждого пользователя
    sesions_per_user = df_dict['visits_df'].groupby('user_id')['session_id'].nunique()
    # Перевод в относительные значения
    session_per_category = session_per_category.div(sesions_per_user, axis='index')
    # Сбрасываем индекс
    session_per_category = session_per_category.reset_index()

    '''Создаем агрегированные признаки количества сессий по временным категориям для каждого пользователя из visits_df'''
    session_per_daytime = pd.pivot_table(df_dict['visits_df'],
                                         index='user_id',
                                         columns='daytime',
                                         values='session_id',
                                         aggfunc='nunique',
                                         fill_value=0)

    # Убираем отображение имени колонок в pivot
    session_per_daytime.columns.name = None
    # Перевод в относительные значения
    session_per_daytime = session_per_daytime.div(sesions_per_user, axis='index')
    # Сбрасываем индекс
    session_per_daytime = session_per_daytime.reset_index()

    '''Создаем агрегированный признак кол-во сессий'''
    sesions_per_user = sesions_per_user.reset_index()
    sesions_per_user = sesions_per_user.rename(columns={'session_id': 'amount_sessions'})

    '''Объединяем все таблицы в одну'''
    result = (df_dict['users_df']
          .merge(df_dict['ads_activity_df'], on='user_id', how='left')
          .merge(df_dict['surf_depth_df'], on='user_id', how='left')
          .merge(df_dict['primary_device_df'], on='user_id', how='left')
          .merge(df_dict['cloud_usage_df'], on='user_id', how='left')
          .merge(session_per_category, on='user_id', how='left')
          .merge(session_per_daytime, on='user_id', how='left')
          .merge(sesions_per_user, on='user_id', how='left'))    

    '''Приводим названия к snake_case'''
    snake_names = []
    for col_name in result.columns:
        col_name = col_name.lower()
        if ' ' in col_name:
            col_name = col_name.replace(' ', '_')         
        snake_names.append(col_name)

    result.columns = snake_names
    # Меняем название столбцов на английские
    result = result.rename(columns={'вечер': 'evening', 'день': 'afternoon', 'ночь': 'night', 'утро': 'morning'})

    # Убираем user_id как бесполезный для обучения модели
    result = result.drop(columns=['user_id'])

    return result
