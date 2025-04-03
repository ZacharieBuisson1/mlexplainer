import numpy as np
import pandas as pd
import sklearn

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures


### Création d'une classe de Data Cleaner ###
class dataCleaner(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Classe qui va nettoyer les données, en créer quelques variables dans le dataframe."""

    def __init__(
        self, numerators, denominators, polynomial_features, strategy="mean"
    ):
        """Initialisation des variables de la classe."""
        self.features = list()
        self.numerators = numerators
        self.denominators = denominators
        self.polynomial_features = polynomial_features

        ### Création d'une stratégie d'imputation ###
        self.imputer = SimpleImputer(strategy=strategy)

        ### Création de variables polynomiales - ici, valeur par défaut à 3 ###
        self.polynomer = PolynomialFeatures(3, include_bias=False)

    def fit(self, X, y=None):
        ### Entrainement des variables du modèle ###
        self.features = X.columns.tolist()

        ### Entrainement de l'imputer ###
        X_imputed = pd.DataFrame(
            self.imputer.fit(X[self.polynomial_features]),
            columns=self.polynomial_features,
        )

        ### Entrainement du polynomer ###
        self.polynomer.fit(X_imputed)

        return self

    def transform(self, X, y=None):
        """Fonction de transforamtion après le fitting."""

        ### Possibilité de créer une copie du dataframe : attention à la taille du dataframe ###
        # X_copy = X.copy()

        #######################################################################
        ### Exemple de traitement qu'il est possible de mettre dans ce code ###
        #######################################################################

        # 1. Nombre d'occurence d'une élément dans le dataframe
        # Exemple :
        # li_multiple_elem = X_copy['NUMERODOSSIER'].value_counts().to_frame().query('NUMERODOSSIER>=2').index.tolist()
        # X_copy['multiple_elem'] = 0
        # X_copy.loc[X_copy['NUMERODOSSIER'].isin(li_multiple_elem), "NUMERODOSSIER"] = 1

        # 2. Faire des différences sur les dates
        # Exemple :
        # feature_date = ['date1', 'date2']
        # for date in feature_date :
        #   X_copy['DELTA_' + date] = (pd.to_datetime(X_copy['DATE_DEMANDE'], format='%Y-%m-%d') - pd.to_datetime(X_copy[date], format = '%Y-%m-%d')).apply(lambda u:u.days)

        # 3. Construction de ratios
        # Exemple :
        # for denominator in self.denominators:
        #   X_copy = X_copy.join(create_all_ratios(X_copy, [c for c in self.numerators if c != denominator], self.denominator))

        # 4. Création des variables quantitatives sur les variables types flag
        # for flag in var_flag:
        #   X_copy[flag] = X_copy[flag].astype(str)

        # 5. Création de CL logiques de variables
        # X_copy[sum_var1_var2] = X_copy[var_1].add(X_copy[var_2])
        # X_copy[ratio_var1_var2] = X_copy[var_1].divide(X_copy[var_2])
        # X_copy[mean_var1_var2] = X_copy[var_1].add(X_copy[var_2])/2

        # 6. Imputation de valeurs manquantes
        # X_imputed = pd.DataFrame(self.imputer.transform(X_copy[self.polynomial_features]), columns=self.polynomial_features)

        # 7. Création de variables pomynomiales
        # X_polynomial = pd.DataFrame(polynomer.transform(X_imputed), columns=polynomer.get_feature_names())
        # X_imputed.columns = [c+'_IMPUTED' for c in X_imputed.columns.tolist()]
        # feature_imputed = X_polynomial.columns.tolist()
        # X_polynomial.drop(self.polynomial_features, axis=1, inplace=True)
        # renamed_polynomial_name = []
        # for c in range(0, len(X_polynomial.columns)):
        #   renamed_polynomial_name.append(X_polynomial.columns[c])
        #   for j in range(0, len(self.polynomial_features)):
        #       renamed_polynomial_name[c] = renamed_polynomial_name[c].replace('x'+str(j), self.polynomial_features[j])
        #       renamed_polynomial_name[c] = renamed_polynomial_name[c].replace('^'+str(j), '_'+str(self.polynomial_features[j]))
        # X_polynomial.columns = list(map(lambda u : u.replace(' ', '_x_'), renamed_polynomial_name))
        # X_copy = X_copy.join(X_polynomial)
        # X_copy = X_copy.join(X_imputed)
        # del X_imputed
        # del X_polynomial


def create_all_ratios(df_ref, numerators, denominator):
    """Création des tous les ratios avec comme numérateur un des numérateur et dénominateur le dénominateur d'usage."""

    ### Confirmation que les colonnes ne soient pas des objets ###
    numerators_ok = [x for x in numerators if df_ref[x].dtype != "o"]

    ### Réalisation de la division avec divide ###
    df_ratios = df_ref[numerators_ok].divide(
        df_ref[denominator].replace(0, np.nan), axis=1
    )

    ### Renommage des colonnes ###
    df_ratios.columns = [
        c + "_ratio_" + denominator for c in df_ratios.columns.tolist()
    ]

    return df_ratios
