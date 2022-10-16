import streamlit as st
import pickle
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from sklearn.feature_selection import f_classif
from sklearn import pipeline, preprocessing, compose
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA, TruncatedSVD
from optbinning import OptimalBinning, BinningProcess
import xgboost as xgb

with open('models/pipe.pkl','rb') as pi:
    pipe = joblib.load(pi)

with open('models/regresión_logistica_model.pkl','rb') as lr:
    log_reg = joblib.load(lr)

with open('models/Arbol_model.pkl','rb') as tr:
    tree = joblib.load(tr)

with open('models/voting_model_model.pkl','rb') as tr:
    voting = joblib.load(tr)


def main():
    st.title('Modelamiento de riesgo crediticio')
    st.sidebar.header('Caracteristicas del prestatario')

    def translate(feature:str, value:str):

        home_ownership = {
            'ALQUILER':'RENT',     'PROPIO':'OWN',
            'HIPOTECA':'MORTGAGE', 'OTROS':'OTHER'
        }

        verification_status = {
            'VERIFICADO': 'Verified',
            'FUENTE VERFICADA': 'Source Verified',
            'NO VERIFICADO': 'Not Verified'
        }

        purpose = {
            'Tarjeta de credito':'credit_card', 'Carro': 'car',
            'Micro empresa': 'small_business', 'Boda':'wedding',
            'Consolidar deuda':'debt_consolidation', 'Mejora del hogar': 'home_improvement',
            'Compra importante': 'major_purchase', 'Salud': 'medical',
            'Traslado': 'moving', 'Vacaciones':'vacation',
            'Casa': 'house','Energia renovable': 'renewable_energy',
            'Educacion':'educational', 'Otros': 'other'
        }

        if feature == 'home_ownership':
            return home_ownership[value]
        elif feature == 'verification_status':
            return verification_status[value]
        elif feature == 'purpose':
            return purpose[value]
        else:
            return 'NO SE PAI'

    def user_input_parameters():
        term = st.sidebar.selectbox('Numero de pagos',(36,60))
        int_rate = st.sidebar.number_input('Tasa de interes')
        grade = st.sidebar.selectbox('Nota asignada por Lending club', ('A','B','C','D','E','F','G'))
        home_ownership = st.sidebar.selectbox('Estado de propiedad', ('ALQUILER','PROPIO','HIPOTECA','OTROS')) #TRADUCIR
        verification_status = st.sidebar.selectbox('Verificacion status', ('VERIFICADO','FUENTE VERFICADA', 'NO VERIFICADO')) #TRADUCIR
        annual_inc = st.sidebar.number_input('Ingreso anual')
        purpose = st.sidebar.selectbox('Proposito del prestamo', ['Tarjeta de credito', 'Carro', 'Micro empresa', 'Boda',
                                                                'Consolidar deuda', 'Mejora del hogar', 'Compra importante',
                                                                'Salud', 'Traslado', 'Vacaciones', 'Casa', 'Energia renovable',
                                                                'Educacion', 'Otros']) #TRADUCIR
        dti = st.sidebar.number_input('DTI')
        inq_last_6mths = st.sidebar.number_input('Numero de consultas en los ultimos 6 meses')
        revol_util = st.sidebar.number_input('tasa de utilizacion de la linea rotatoria')
        initial_list_status = st.sidebar.selectbox('Estado inicial de la lista del prestamo', ('f','w')) #QUESESO
        total_rec_int = st.sidebar.number_input('Intereses recibidos hasta la fecha')
        tot_cur_bal = st.sidebar.number_input('Saldo total de todas las cuentas')
        total_rev_hi_lim = st.sidebar.number_input('Limite del credito')
        mths_since_issue_d = st.sidebar.number_input('Meses desde que se hizo el prestamo')
        mths_since_last_credit_pull_d = st.sidebar.number_input('Meses desde que saco credito para este prestamo')
        #bad_loan = st.sidebar.selectbox('Estado del prestamo', ('PAGO', 'INCOBRABLE','VIGENTE', 'MORA', 
        #                                'TARDE (31-120 dias)','EN PERIODO DE GRACIA', 'TARDE (16-30 dias)'))#TRADUCIR
        try:
            home_ownership = translate('home_ownership', home_ownership)
            verification_status = translate('verification_status', verification_status)
            purpose = translate('purpose', purpose)
        except AssertionError as error:
            print(error)
            print('NO SE PUDO EJECUTAR TRANSLATE')

        data = {
            'term':term,
            'int_rate': int_rate,
            'grade': grade,
            'home_ownership': home_ownership,
            'annual_inc': annual_inc,
            'verification_status':verification_status,
            'purpose': purpose,
            'dti': dti,
            'inq_last_6mths': inq_last_6mths,
            'revol_util': revol_util,
            'initial_list_status': initial_list_status,
            'total_rec_int': total_rec_int,
            'tot_cur_bal': tot_cur_bal,
            'total_rev_hi_lim': total_rev_hi_lim,
            'mths_since_issue_d':mths_since_issue_d,
            'mths_since_last_credit_pull_d':mths_since_last_credit_pull_d,
        }

        features = pd.DataFrame(data, index =[0])
        return features
    df = user_input_parameters()
    st.subheader('User Input Parameters')
    st.write(df)
    #new_df = pipe.transform(df)
    if st.button('RUN'):
        st.success(voting.predict(df))
if __name__ == '__main__':
    main()