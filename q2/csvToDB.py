import pandas as pd
import sqlite3
import os
import numpy as np

def users_table_create():
    return "CREATE TABLE users (user_id PRIMARY KEY, gender_cd, age, foreigner_yn, os_type)"

def dutchpay_claim_tabel_create():
    return "CREATE TABLE dutchpay_claim (claim_id PRIMARY KEY, claim_at, claim_user_id REFERENCES users(user_id))"

def dutchpay_claim_detail_table_create():
    return "CREATE TABLE dutchpay_claim_detail \
        (\
            claim_detail_id PRIMARY KEY, \
            claim_id REFERENCES dutchpay_claim(claim_id), \
            recv_user_id REFERENCES users(user_id), \
            claim_amount, \
            send_amount, \
            status\
        )"

def a_payment_trx():
    return "CREATE TABLE a_payment_trx(id PRIMARY KEY, transaction_id, transacted_at, payment_action_type, user_id REFERENCES users(user_id), amount)"


def create_tables():
    db = "db.sqlite3"
    conn = sqlite3.connect(db)
    conn.text_factory = str  # allows utf-8 data to be stored
    cur = conn.cursor()

    cur.execute('PRAGMA foreign_keys = ON')

    sql = "DROP TABLE IF EXISTS dutchpay_claim_detail"
    cur.execute(sql)

    sql = "DROP TABLE IF EXISTS users"
    cur.execute(sql)

    sql = "DROP TABLE IF EXISTS dutchpay_claim"
    cur.execute(sql)

    sql = "DROP TABLE IF EXISTS a_payment_trx"
    cur.execute(sql)

    cur.execute(users_table_create())
    cur.execute(dutchpay_claim_tabel_create())
    cur.execute(dutchpay_claim_detail_table_create())
    cur.execute(a_payment_trx())

    conn.commit()


def make_db(csv_filename):
    # sqlite connect
    db = "db.sqlite3"
    conn = sqlite3.connect(db)
    conn.text_factory = str  # allows utf-8 data to be stored
    cur = conn.cursor()

    table_name = csv_filename.split('/')[-1].split('.')[0]
    print(table_name)
    data = pd.read_csv(csv_filename)

    # insert data
    for idata in data.itertuples():
        tmp_data = []
        columns = []
        for i, tmp in enumerate(idata[1:]):
            if type(tmp) == str:
                tmp_data.append("'" + tmp + "'")
                columns.append(data.columns.tolist()[i])
            elif not(np.isnan(tmp)):
                tmp_data.append(str(tmp))
                columns.append(data.columns.tolist()[i])
        
        sql = "INSERT INTO %s (%s) VALUES (%s)" % (table_name, ", ".join(columns), ", ".join([str(i) for i in tmp_data]))
        print(sql)
        cur.execute(sql)

    conn.commit()
    return table_name

def get_reward_user_id():
    # sqlite connect
    db = "db.sqlite3"
    conn = sqlite3.connect(db)
    conn.text_factory = str  # allows utf-8 data to be stored
    cur = conn.cursor()

    sql = "\
    SELECT user_id \
    FROM (SELECT user_id, payment_action_type, transacted_at, \
        sum(CASE WHEN payment_action_type = 'PAYMENT' THEN \
                    amount \
                ELSE \
                    amount * -1 \
                END) AS sum_amount \
        FROM a_payment_trx \
        WHERE transacted_at >= '2019-12' \
        AND ((payment_action_type = 'PAYMENT' AND transacted_at <= '2020-01') \
        OR (payment_action_type != 'PAYMENT' AND transacted_at <= '2020-03')) \
        GROUP BY user_id) \
    WHERE sum_amount > 10000;"  

    sql = "\
    CREATE TEMPORARY TABLE sum_amount AS SELECT user_id, payment_action_type, transacted_at, \
        sum(CASE WHEN payment_action_type = 'PAYMENT' THEN \
                    amount \
                ELSE \
                    amount * -1 \
                END) AS sum_amount \
    FROM a_payment_trx \
    WHERE transacted_at >= '2019-12' \
        AND ((payment_action_type = 'PAYMENT' AND transacted_at <= '2020-01') \
        OR (payment_action_type != 'PAYMENT' AND transacted_at <= '2020-03')) \
    GROUP BY user_id;" 
    cur.execute(sql)

    sql = "SELECT user_id FROM sum_amount WHERE sum_amount > 10000;"
    cur.execute(sql)

    result = cur.fetchall()
    print('user_id :', result)

    conn.commit()

if __name__ == '__main__':
    # create tabel
    create_tables()
    # insert data
    make_db('../csv_data/users.csv')
    make_db('../csv_data/a_payment_trx.csv')
    make_db('../csv_data/dutchpay_claim.csv')
    make_db('../csv_data/dutchpay_claim_detail.csv')

    get_reward_user_id()

    