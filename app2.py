import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns

def main():
    st.set_page_config(page_title="Logistic Regression App")
    st.title("Logistic Regression App")
    
    file = st.file_uploader("Upload CSV", type="csv")
    
    if file is not None:
        df = pd.read_csv(file)
        target = st.selectbox("Select Target Column", df.columns)
        features_choise = st.multiselect("Select Features Columns", df.columns)
        learning_rate=st.slider("Select Learning Rate",min_value=0.01, max_value=1.0, value=0.1, step=0.01)
        epoch=st.slider("Select Number of Epochs",min_value=3000, max_value=10000, step=1)


        
        features = df[features_choise]
        target_values = df[target]
        # Дробление на обучающую и тестовую части
        X_train, X_test, y_train, y_test = train_test_split(features, target_values, test_size=0.2, random_state=42)
        # Нормализация всех столбцов, кроме целевого   
        scaler = StandardScaler()     
        X_train=pd.DataFrame(scaler.fit_transform(X_train), columns=features.columns)
        X_test=pd.DataFrame(scaler.fit_transform(X_test), columns=features.columns)
        y_train = y_train.values
        y_test = y_test.values
        

        # Объявление класса
        class LogReg:
            def __init__(self, learning_rate, n_inputs):
                self.learning_rate = learning_rate
                self.n_inputs = n_inputs
                self.coef_ = np.random.normal(size=(n_inputs))
                self.intercept_ = np.random.normal()

            def sigmoida(self, X):
                return 1 / (1 + np.exp(-X))
                
            def fit(self, X, y):
                for i in range(epoch):
                    y_pred = self.sigmoida(np.dot(X, self.coef_) + self.intercept_)

                    error=(y_pred-y)
                    w0_grad=np.mean(error)
                    w_grad=np.dot(X.T,error)/len(y)
                    self.coef_ -= self.learning_rate * w_grad 
                    self.intercept_ -= self.learning_rate * w0_grad     
                    
            def predict(self, X):
                return np.round(self.sigmoida(np.dot(X, self.coef_) + self.intercept_))
            
            def score(self, X, y):
                return accuracy_score(y, np.round(self.sigmoida(np.dot(X, self.coef_) + self.intercept_)))
        
        # Вызов класса
        lr = LogReg(learning_rate,features.shape[1])
        lr.fit(X_train, y_train)
        personal=lr.predict(X_test)
        weights = dict(zip(features.columns, lr.coef_))

        # Описание весов
        st.write("Weights:")
        for col, weight in weights.items():
            st.write(f"{col}: {weight:.4f}")

        # Выбор формы графика
        fig_type = st.selectbox("Select Chart Type", ["Scatter Plot", "Bar Plot", "Line Plot"])
        if fig_type == "Scatter Plot":          
            selected_features = st.multiselect("Select Features", features.columns)
            if len(selected_features) >= 2:
                
                # Используем только выбранные пользователем признаки для построения скаттерплота
                fig, ax = plt.subplots()
                sns.scatterplot(data=df, x=selected_features[0], y=selected_features[1], hue=target, ax=ax)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.warning("Please select at least 2 features to plot a scatter plot.")

        elif fig_type == "Bar Plot":
            feature = st.selectbox("Select Feature", features.columns)
            df_plot = pd.concat([features[feature], target_values], axis=1)
            df_plot.groupby(target)[feature].mean().plot(kind="bar")
            st.pyplot()
        else:
            feature = st.selectbox("Select Feature", features.columns)
            sns.lineplot(data=df, x=feature, y=target)
            st.pyplot()

if __name__ == "__main__":
    main()
    
