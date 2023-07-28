import pandas as pd
import sys
import numpy as np
import os
from google.cloud import bigquery
from flask import Flask, render_template, request, escape

app = Flask(__name__)

#ONLY FOR LOCAL USE
#CREDS = "C:\\Users\\TK\Desktop\\msds434-module6-dba29024def3.json"
#client = bigquery.Client.from_service_account_json(json_credentials_path=CREDS,project='msds434-module6')
client = bigquery.Client(project='msds434-module6')

dset = 'module6'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_base_model')
def form():
    return render_template('form.html')

@app.route('/data', methods = ['POST', 'GET'])
def data():
    if request.method == 'POST':
        form_data = request.form
        age = form_data['Age']
        workclass = str(form_data['Workclass'])
        marital_status = str(form_data['marital_status'])
        education_num = form_data['education_num']
        occupation = str(form_data['Occupation'])
        hours_per_week = form_data['hours_per_week']
        # submit the input 
        records = [
            {'age':age,
             'workclass':workclass,
             'marital_status':marital_status,
             'education_num':education_num,
             'occupation':occupation,
             'hours_per_week':hours_per_week}
        ]

        df = pd.DataFrame(
            records,
            columns = [
                'age',
                'workclass',
                'marital_status',
                'education_num',
                'occupation',
                'hours_per_week'
            ]
        )

        job_config = bigquery.LoadJobConfig(
        
            schema = [
                bigquery.SchemaField("age", "INTEGER"),
                bigquery.SchemaField("workclass", "STRING"),
                bigquery.SchemaField("marital_status", "STRING"),
                bigquery.SchemaField("education_num", "INTEGER"),
                bigquery.SchemaField("occupation", "STRING"),
                bigquery.SchemaField("hours_per_week", "INTEGER"),
            ],
            autodetect=False,
            source_format=bigquery.SourceFormat.CSV,
            write_disposition="WRITE_TRUNCATE" #comment this out to append
        )
        job = client.load_table_from_dataframe(
            df, 'msds434-module6.module6.input',job_config=job_config
        )  

        job.result()

        # do the predict
        query = """
        SELECT *
        FROM ML.PREDICT(MODEL module6.model1,
        (SELECT *
        FROM
        `msds434-module6.module6.input`),
        STRUCT(0.5 AS threshold)
        )
        """

        df = client.query(query).to_dataframe()
        df2 = pd.DataFrame(df.explode('predicted_income_bracket_probs')['predicted_income_bracket_probs'])
        df2 = df2['predicted_income_bracket_probs'].apply(pd.Series)
        pivoted = df2.pivot_table(index=df2.index,columns='label',values='prob',aggfunc=np.mean)
        pivoted.rename(columns={ pivoted.columns[0]: "prob_<=50K" ,pivoted.columns[1]: "prob_>50k" }, inplace = True)
        merged = df.drop('predicted_income_bracket_probs',axis=1)
        merged = merged.join(pivoted)
        merged
        return render_template('data.html',tables=[merged.to_html(max_rows=20,classes='data')], titles=['predictions'])
    
 
 
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
