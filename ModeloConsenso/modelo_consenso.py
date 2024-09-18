import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from joblib import load


def ModeloConsenso(smiles):

    # Calcular huella de morgan
    mol = Chem.MolFromSmiles(smiles)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    fingerprint_list = list(fingerprint)
    finger_array = np.array(fingerprint_list).reshape(1, -1)

    # Crear un DataFrame con nombres de columnas ficticios
    columns = [f'{i}' for i in range(finger_array.shape[1])]
    finger_df = pd.DataFrame(finger_array, columns=columns)

    # Importar modelos
    SVM_model = load("GuardadoDeModelos/modelo_SVM.joblib")
    RF_model = load("GuardadoDeModelos/modelo_RF.joblib")
    XGB_model = load("GuardadoDeModelos/modelo_XGBoost.joblib")

    # Calcular predicciones
    y1 = SVM_model.predict(finger_df)
    y1_prob = SVM_model.predict_proba(finger_df)

    y2 = RF_model.predict(finger_df)
    y2_prob = RF_model.predict_proba(finger_df)

    y3 = XGB_model.predict(finger_df)
    if y3 == 0:
        y3 = ["BBB+"]
    else:
        y3 = ["BBB-"]
    y3_prob = XGB_model.predict_proba(finger_df)

    consensus_probability = ((y1_prob[0][0]+y2_prob[0][0]+y3_prob[0][0])/3)
    clase = None

    if (y1 == ["BBB+"]) and (y2 == ["BBB+"]) and (y3 == ["BBB+"]):
        clase = "BBB+"
    else:
        clase = "BBB-"

    return clase, consensus_probability
