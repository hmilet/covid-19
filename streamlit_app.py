import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import covid19_dataviz as viz
import pickle
import covid19_cnn_utils as cnn
import covid19_efficientnet_utils as eff
from tensorflow.keras import models
import covid19_main_gradcam_densenet as gc_densenet

@st.cache_resource
def get_data():
    with open('dict_to_df.pickle', 'rb') as handle:
        dict_to_df = pickle.load(handle)
    with open('class_image_arrays.pickle', 'rb') as handle:
        class_image_arrays = pickle.load(handle)
    with open('class_mask_arrays.pickle', 'rb') as handle:
        class_mask_arrays = pickle.load(handle)
    pixel_stats_df = viz.get_pixel_stats(class_image_arrays)
    image_level_stats_df = viz.get_image_level_pixel_stats(class_image_arrays)
    return dict_to_df, class_image_arrays, class_mask_arrays, pixel_stats_df, image_level_stats_df

@st.cache_resource
def get_models():
    with open('eff_model.pickle', 'rb') as handle:
        eff_model = pickle.load(handle)
    with open('model.pickle', 'rb') as handle:
        cnn_model = pickle.load(handle)
    with open('xtest.pickle', 'rb') as handle:
        X_test = pickle.load(handle)
    with open('ytest.pickle', 'rb') as handle:
        y_test = pickle.load(handle)
    dense_model = models.load_model("densenet121_final.keras")
    return eff_model, cnn_model, dense_model, X_test, y_test

dict_to_df, class_image_arrays, class_mask_arrays, pixel_stats_df, image_level_stats_df = get_data()

eff_model, cnn_model, dense_model, X_test, y_test = get_models()

dict_class = {
    0 : "COVID"
    , 1 : "Lung Opacity"
    , 2 : "Normal"
    , 3 : "Pneumonia"
}

st.sidebar.title('Sommaire')

pages = ['Contexte'
         , 'Exploration de données'
         , 'Prétraitement des données'
         , 'Modèles explorés'
         , 'Prédiction'
         , 'Résumé']

page = st.sidebar.radio('Aller vers', pages)

if page == pages[0] :
    ### Contexte

    st.title("Analyse de radiographies pulmonaires COVID-19")
    st.header("Projet de classification")

    st.markdown(
        """
        ### Introduction
        <div style="text-align: justify;">
        La pandémie de COVID-19 a fortement sollicité les systèmes de santé, rendant nécessaire une évaluation rapide de l'état pulmonaire des patients. Économique, rapide et accessible, la radiographie thoracique permet de détecter les anomalies respiratoires liées au virus.
        <br>
        <br>
        En s'appuyant sur les récents progrès de l'intelligence artificielle et de la vision par ordinateur, ce projet explore l'utilisation du deep learning pour automatiser la classification de ces radiographies médicales.
        </div>
        <br>
        <br>
    """
        , unsafe_allow_html=True
    )

    st.markdown(
        """
        ### Enjeu métier
        <div style="text-align: justify;">
        Développer un outil basé sur l'imagerie X pour détecter classifier les pathologies pulmonaires : absence, pneumonies et COVID-19. L'objectif est de minimiser les erreurs de diagnostic, notamment pour le COVID-19, afin d'optimiser le tri et la prise en charge des patients par les professionnels de santé.
        </div>
        <br>
        <br>
    """
        , unsafe_allow_html=True
    )

    st.markdown(
        """
        ### Jeu de données et bibliographie
        <div style="text-align: justify;">
        Les données utilisées dans ce projet proviennent du dataset kaggle accessible
        <a href ='https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database'>ici</a>.
        <br>
        <br>
        Articles ayant permis la création du jeu de données:

        - [M. E. H. Chowdhury et al., 2020](https://doi.org/10.1109/ACCESS.2020.3010287).
        - [T. Rahman et al., 2021](https://doi.org/10.1016/j.compbiomed.2021.104319).
        </div>
        <br>
        <br>
    """
        , unsafe_allow_html=True
    )

if page == pages[1]:
    ### Exploration de données

    st.title("Exploration du jeu de données")

    st.header("Structure des données")

    st.markdown(
        """
        <div style="text-align: justify;">
        Le jeu de données "COVID-19 Radiography Database" rassemble des radiographies pulmonaires réparties en quatre catégories : COVID-19, pneumonie virale, lung opacity et cas sains. Chaque image est accompagnée d'un masque de segmentation délimitant précisément la zone des poumons.
        </div>
        <br>
        <br>
    """
        , unsafe_allow_html=True
    )

    df = pd.DataFrame(
        {
        "Caracéristiques": ["Nombre images", "Nombre classes", "Volume dataset", "Résolution images", "Résolution masques"],
        "Valeurs": ["21165", "4", "1.15 GB", "299 x 299", "256 x 256"]
        }
    )

    st.dataframe(df, hide_index = True, use_container_width = True)

    class_name = 'Normal'

    if "selected_button" not in st.session_state:
        st.session_state.selected_button = "Normal" # Option par défaut

    options = ["Normal", "COVID", "Lung Opacity", "Viral Pneumonia"]

    # Affichage des 4 boutons côte à côte
    cols = st.columns(len(options))

    for i, option in enumerate(options):
        with cols[i]:
            # Le bouton prend le style "primary" SEULEMENT si son nom correspond à l'option active
            is_selected = (st.session_state.selected_button == option)
            
            if st.button(
                option, 
                key=f"btn_{i}", 
                width = 'stretch',
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.selected_button = option
                class_name = option
                st.rerun() # Force la mise à jour immédiate de l'affichage

    img_idx = st.slider("Index de l'image :", min_value = 0, max_value = 10, step = 1)

    fig, ax = viz.display_image_mask_pair(
        class_image_arrays,
        class_mask_arrays,
        class_name=class_name,
        image_index=img_idx
    )

    st.pyplot(fig)



    st.header("Distribution des classes")

    df = pd.DataFrame.from_dict(dict_to_df)

    g = sns.catplot(
        data=df, kind="bar",
        x='category', y='number_img', hue = 'category',
        errorbar="sd", palette="pastel", alpha=.6, height=6
    )

    g.despine(left=True)
    g.set_axis_labels("", "Nombre d'images")

    last_i = 0

    for ax in g.axes.flat:

        for i, bar in enumerate(ax.patches):
            height = bar.get_height()

            row = df.iloc[i + last_i]

            total_val = row["number_img"]
            unique_val = row["unique_number_img"]

            # Format the label
            label_text = f"{total_val}\n({unique_val} unique)"

            ax.text(
                x=bar.get_x() + bar.get_width() / 2,
                y=height,
                s=label_text,
                ha="center",
                va="bottom",
                fontsize=9
            )
        
        last_i = i + 1

        st.pyplot(g)

    st.markdown(
        """
        <div style="text-align: justify;">
        Les classes sont déséquilibrées, une augmentation de données a donc été réalisée sur les trois classes minoritaires pour atteindre l'équilibre avec les paramètres suivants : 
        
        - RandomRotation (± 15%)
        - RandomZoom (± 10%)
        - RandomContrast (± 10%)
        </div>
        <br>
        <br>
        """
        , unsafe_allow_html=True
    )

    st.header("Homogénéité des images")

    st.markdown(
        """
        <div style="text-align: justify;">
        On s'intéresse à l'homogénéité des images entre les différentes classes par rapport à leur luminosité (moyenne de la valeur des pixels) et leur constraste (écart-type de la valeur des pixels).
        </div>
        <br>
        <br>
    """
        , unsafe_allow_html=True
    )


    fig2, ax2 = plt.subplots(figsize=(8, 5))

    sns.boxplot(
        data=image_level_stats_df,
        x="class",
        y="pixel_mean",
        ax=ax2,
        palette="pastel"
    )

    ax2.set_title("Intensité moyenne des pixels par classe")
    ax2.set_xlabel("Classe")
    ax2.set_ylabel("Intensité moyenne")
    ax2.tick_params(axis='x', rotation=20)

    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        data=image_level_stats_df,
        x="class",
        y="pixel_std",
        ax=ax3,
        palette="pastel"
    )

    ax3.set_title("Contraste moyen des pixels par classe")
    ax3.set_xlabel("Classe")
    ax3.set_ylabel("Intensité moyenne")
    ax3.tick_params(axis='x', rotation=20)

    st.pyplot(fig3)

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=image_level_stats_df,
        x="pixel_mean",
        hue="class",
        bins=40,
        kde=True,
        element="step"
        , palette = 'pastel'
        , ax = ax4
    )

    ax4.set_title("Distribution de la luminosité moyenne par image")
    ax4.set_xlabel("Luminosité moyenne")
    ax4.set_ylabel("Nombre d'images")

    st.pyplot(fig4)

    fig5, ax5 = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=image_level_stats_df,
        x="pixel_std",
        hue="class",
        bins=40,
        kde=True,
        element="step"
        , palette = 'pastel'
        , ax = ax5
    )

    ax5.set_title("Distribution du contraste par image")
    ax5.set_xlabel("Écart-type des pixels")
    ax5.set_ylabel("Nombre d'images")

    st.pyplot(fig5)

if page == pages[2]:
    ### Prétraitement des données

if page == pages[3]:
    ### Modèles explorés


if page == pages[4]:
    ### Prédiction

    st.title("Prédiction")

    st.header("Modèle à utiliser:")

    model_dict = {'CNN' : cnn_model
                  , 'DenseNet121' : dense_model
                  , 'EfficientNetB2' : eff_model}

    if "selected_button" not in st.session_state:
        st.session_state.selected_button = "CNN" # Option par défaut
        model = model_dict['CNN']

    options = ["CNN", "DenseNet121", "EfficientNetB2"]

    # Affichage des 4 boutons côte à côte
    cols = st.columns(len(options))

    for i, option in enumerate(options):
        with cols[i]:
            # Le bouton prend le style "primary" SEULEMENT si son nom correspond à l'option active
            is_selected = (st.session_state.selected_button == option)
            
            if st.button(
                option, 
                key=f"btn_{i}", 
                width = 'stretch',
                type="primary" if is_selected else "secondary"
            ):
                st.session_state.selected_button = option

                st.rerun() # Force la mise à jour immédiate de l'affichage


    model = model_dict[st.session_state.selected_button]

    st.header("Image à prédire:")

    fig_x, ax_x = plt.subplots(figsize=(8, 5))


    img_idx = st.slider("Index de l'image :", min_value = 0, max_value = 15, step = 1)

    plt.imshow(X_test[img_idx], cmap = 'Grays_r')

    ax_x.set_title(f"Classe réelle : {dict_class[y_test[img_idx]]}")
    ax_x.set_axis_off()

    st.pyplot(fig_x)

    single_img = X_test[img_idx]

    img_batch = np.expand_dims(single_img, axis=0)

    pred = model.predict(img_batch)
    class_pred = np.argmax(pred, axis=1)
    max_val = np.max(pred)

    st.header(f"Prédiction du modèle :")
    st.subheader(f"{dict_class[class_pred[0]]} ➡️ {max_val:.2%} de confiance")

    if st.session_state.selected_button == "EfficientNetB2":
        fig_mod, class_report = eff.evaluate_efficientnet_model(
            model,
            X_test,
            y_test,
            img_idx
        )

        df = pd.DataFrame.from_dict(class_report).transpose().drop(columns = ['support'])

        df.rename(index = {
            '0' : "COVID"
            , '1' : "Lung Opacity"
            , '2' : "Normal"
            , '3' : "Pneumonia"
            }
        , inplace=True)

        st.pyplot(fig_mod)

        st.header('Performances globales du modèle :')

        st.dataframe(df.style.format("{:.2f}"), use_container_width = True)

    if st.session_state.selected_button == "DenseNet121":
        #fig_mod, class_report = gc_densenet.explain(dense_model, X_test, y_test, img_idx)
        fig_mod = gc_densenet.explain(dense_model, X_test, y_test, img_idx)
        st.pyplot(fig_mod)

        # df = pd.DataFrame.from_dict(class_report).transpose().drop(columns = ['support'])
        
        # df.rename(index = {
        #             '0' : "COVID"
        #             , '1' : "Lung Opacity"
        #             , '2' : "Normal"
        #             , '3' : "Pneumonia"
        #             }
        # , inplace=True)
        df = pd.read_csv('densenet121_final_classification_report.csv',index_col=0)
        st.header('Performances globales du modèle :')
        
        st.dataframe(df.style.format("{:.2f}"), use_container_width = True, width='stretch')

if page == pages[5] :
    ### Résumé

    st.write('bidule')