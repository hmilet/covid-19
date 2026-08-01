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
         , 'Conclusion']

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

    st.dataframe(df, hide_index = True, width = 'stretch')

    #class_name = 'COVID'
    #print("begin class_name:", class_name)
    if "selected_button" in st.session_state and st.session_state.selected_button in ["CNN", "DenseNet121", "EfficientNetB2"]:
        del st.session_state.selected_button

    if "selected_button" not in st.session_state:
        st.session_state.selected_button = "Normal" # Option par défaut
    class_name = st.session_state.selected_button

    options_explo = ["Normal", "COVID", "Lung Opacity", "Viral Pneumonia"]

    # Affichage des 4 boutons côte à côte
    cols = st.columns(len(options_explo))

    def set_class(name):
            st.session_state.selected_button = name

    for i, option in enumerate(options_explo):
        with cols[i]:
            # Le bouton prend le style "primary" SEULEMENT si son nom correspond à l'option active
            is_selected = (st.session_state.selected_button == option)
            #print("option:", option)
            
            #if st.button(
            st.button(
                option, 
                key=f"btn_{i}", 
                width = 'stretch',
                type="primary" if is_selected else "secondary",
                on_click=set_class,
                args=(option,)
            )#:
                #st.session_state.selected_button = option
                #class_name = option
                #print("Button ------> class_name:", class_name)
                #st.rerun() # Force la mise à jour immédiate de l'affichage

    img_idx = st.slider("Index de l'image :", min_value = 0, max_value = 10, step = 1,
                        key="img_idx_pred")  # clé fixe, indépendante du modèle choisi)
    #print("option:", option)
    #print("avant affichage--> class_name:", class_name)
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
        Les classes sont déséquilibrées : un traitement spécifique pour prendre en compte cette problématique devra donc être mis en place.
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
        st.title("Prétraitement des données")

        st.header("Gestion des doublons")

        st.markdown(
            """
            <div style="text-align: justify;">
            En comparant les images, on constate que l'on a des doublons au pixel près selon la distribution suivante : 

            - COVID : <span style="color: #4C72B0;">3616</span> images dont <span style="color: #4C72B0;">3565</span> uniques
            - Lung Opacity : <span style="color: #DD8452;">6012</span> images dont <span style="color: #DD8452;">6012</span> uniques
            - Normal : <span style="color: #55A868;">10192</span> images dont <span style="color: #55A868;">10191</span> uniques
            - Viral Pneumonia : <span style="color: #C44E52;">1345</span> images dont <span style="color: #C44E52;">1338</span> uniques

            Un traitement pour supprimer ces doublons est donc réalisé et également appliqué sur les mêmes index concernant les masques.
            </div>
            <br>
            <br>
            """
            , unsafe_allow_html=True
        )

        st.header("Redimensionnement des images")

        df = pd.DataFrame(
            {
            "Caracéristiques": ["Résolution images", "Résolution masques"],
            "Valeurs": ["299 x 299", "256 x 256"]
            }
        )

        st.markdown(
            """
            <div style="text-align: justify;">
            Comme vu précédemment, les masques et les images ne sont pas à la même résolution : 
            </div>
            <br>
            """
            , unsafe_allow_html=True
        )

        st.dataframe(df, hide_index = True, width = 'stretch')

        st.markdown(
            """
            <div style="text-align: justify;">
            Pour pouvoir les utiliser conjointement et ainsi ne s'intéresser qu'aux zones d'intérêt, des techniques de mise à l'échelle (<i>resize</i>) ont dû être mises en place. 
            Deux approches ont été testées : 
            <br>
            <br>

            - Upsizing des masques (256 x 256 -> 299 x 299) :

            Pour l'upsizing, l'approche retenue est une interpolation par un algorithme Nearest ; pour déterminer la valeur d'un pixel dans l'image agrandie, elle va simplement chercher le pixel le plus proche dans l'image d'origine et copier sa valeur exacte et ne garde donc que des pixels noirs ou blancs (pas de gris).

            - Downsizing des images (299 x 299 -> 256 x 256) :

            L'approche retenue pour le downsizing est par une interpolation utilisant un algorithme Lanczos ; bien qu'un peu plus gourmand d'un point de vue ressource, il est globalement meilleur pour éviter de perdre des informations (pas de "flou" introduit par rapport à la méthode par défaut qui est "nearest").
            </div>
            <br>
            <br>
            """
            , unsafe_allow_html=True
        )

        st.header("Data Augmentation")

        st.markdown(
            """
            <div style="text-align: justify;">
            Pour palier au déséquilibre des classes identifié précédemment, une augmentation de données a été réalisée sur les trois classes minoritaires, pour atteindre l'équilibre.
            Les paramètres suivants ont été retenus : 
        
            - RandomRotation (± 15%)
            - RandomZoom (± 10%)
            - RandomContrast (± 10%)

            Aucun flip n'a été effectué : les poumons ne peuvent pas se retrouver à l'envers sur l'imagerie (pas de flip vertical), et les deux poumons sont asymétriques à cause de la présence du coeur côté gauche (pas de flip horizontal)
            De même, aucune déformation élastique ou cropping n'ont été réalisés pour conserver les proportions anatomiques.
            </div>
            """
            , unsafe_allow_html=True
        )        


        col1, col2 = st.columns(2)

        with col1:
            st.image("X_train_base.png", caption = "Avant data augmentation",  width = 'stretch')

        with col2:
            st.image("X_train_aug.png", caption = "Après data augmentation",  width = 'stretch')

        st.header("Gestion des outliers")

        st.markdown(
            """
            <div style="text-align: justify;">
            Lors de l'exploration du jeu de données, certaines images ont été identifiées comme étant très sombre ou très claires. Une option a été implémentée pour permettre d'entraîner les modèles avec ou sans ces images qui sont assimilables à des valeurs extrêmes (<i>outliers</i>).

            <br>
            <br>
            Pour cela, l'approche choisie a été la sélection de ces outliers via un seuil (1%).
            </div>
            <br>
            <br>
            """
            , unsafe_allow_html=True
        )        

        st.image("outliers_dark.png", width = 'stretch')

        st.image("outliers_light.png", width = 'stretch')


if page == pages[3]:
    ### Modèles explorés
    options_mod = ["Random Forest", "SVM", "CNN", "DenseNet121", "EfficientNetB2"]

    st.title('Modèles explorés')

    model = st.menu_button(label = 'Modèle', options = options_mod)

    if model == "Random Forest":
        st.header('Random Forest :')
        st.image('https://datasciencedojo.com/wp-content/uploads/2024/08/random-forest-algorithm-random-forest.webp',  width = 'stretch')
        st.markdown(
            """
            <div style="text-align: justify;">
            Pour rester sur des temps d'entraînement acceptables sans pour sacrifier trop d'information, une <b>PCR</b> a été effectuée sur <b>500 composants (90% de la variance)</b>.
            Pour garantir une distribution des classes dans chaque plis de données, un <b>K-Fold stratifié</b> a ensuite été appliqué pour l'entraînement.
            Enfin, plusieurs combinaisons d'hyperparamètres ont été testés via un <b>RandomGridSearchCV</b>.
            <br>
            <br>
            Les hyperparamètres retenus pour la meilleure itération sont les suivants:

            <br>
            <br>

            - n_estimators = 300
            - min_samples_split = 2
            - min_samples_leaf = 2
            - max_features = 0.2
            - max_depth = 40

            </div>
            <br>
            <br>
            """
            , unsafe_allow_html=True
        ) 
        df = pd.DataFrame(
            {
            "Classe": ["COVID", "Macro avg"]
            , "Recall": ["0.31","0.66"]
            , "F1-score" : ["0.41","0.65"]
            }
        )
        st.subheader('Résultats obtenus sur ce modèle:')    
        st.dataframe(df, hide_index = True, width = 'stretch')       

    elif model == "SVM":
        st.header('SVM :')
        st.image('https://fr.mathworks.com/discovery/support-vector-machine/_jcr_content/thumbnail.adapt.1200.medium.jpg/1607680863365.jpg',  width = 'stretch')
        st.markdown(
            """
            <div style="text-align: justify;">
            Comme pour la Random Forest, une <b>PCR</b> a été effectuée sur <b>150 composants (80% de la variance)</b>.
            Pour garantir une distribution des classes dans chaque plis de données, un <b>K-Fold stratifié</b> a ensuite été appliqué pour l'entraînement.
            Enfin, plusieurs combinaisons d'hyperparamètres ont été testés via un <b>GridSearchCV</b>.
            <br>
            <br>
            Les hyperparamètres retenus pour la meilleure itération sont les suivants:

            <br>
            <br>
            
            - C = 2
            - gamma = scale
            - kernel = rbf


            </div>
            <br>
            <br>
            """
            , unsafe_allow_html=True
        ) 
        df = pd.DataFrame(
            {
            "Classe": ["COVID", "Macro avg"]
            , "Recall": ["0.44","0.72"]
            , "F1-score" : ["0.50","0.73"]
            }
        )
        st.subheader('Résultats obtenus sur ce modèle:')    
        st.dataframe(df, hide_index = True, width = 'stretch')       

    elif model == "CNN":
        st.header('CNN :')
        st.image('https://miro.medium.com/0*YVT_vA0cgiwkKkPX.png',  width = 'stretch')

        st.markdown(
            """
            <div style="text-align: justify;">
            Toujours dans la même optique d'éviter des temps d'apprentissage trop longs, une <b>fonction de rappel (callback)</b> a été mise en place basée sur un <b>EarlyStopping avec une patience à 30</b> et un <b>suivi de la perte de validation</b> en récupérant les meilleurs poids.
            Plusieurs batch sizes ont été testées empiriquement pour voir ce que la machine utilisée pouvait supporter. 
            Les tests ont également été réalisés sur différentes tailles d'images (via des couches de resize dans le CNN). 
            Enfin, les données d'entraînement ont été séparées en <b>80% de données d'entraînement et 20% de données de validation</b>.
            <br>
            <br>
            Ce modèle dit “baseline” est constitué des couches suivantes :

            <br>
            <br>

            - Couche de rescaling
            - Couches de resize (optionnelles)
            - Couche de convolution, kernel size(3,3) avec 32 neurones, activation ReLU
            - Couche de MaxPooling avec pool size (2,2)
            - Couche de convolution, kernel size (3,3) avec 64 neurones, activation ReLU
            - Couche de MaxPooling avec pool size (2,2)
            - Couche de convolution, kernel size (3,3) avec 128 neurones, activation ReLU
            - Couche de MaxPooling avec pool size (2,2)
            - Couche Flatten
            - Couche Dense avec 128 neurones, activation ReLU
            - Couche Dropout 50% permettant d’éviter l’overfitting
            - Couche Dense à 4 neurones, activation Softmax

            </div>
            <br>
            <br>

            """
            , unsafe_allow_html=True
        ) 
        df = pd.DataFrame(
            {
            "Classe": ["COVID", "Macro avg"]
            , "Recall": ["0.49","0.77"]
            , "F1-score" : ["0.56","0.78"]
            }
        )
        st.subheader('Résultats obtenus sur ce modèle:')    
        st.dataframe(df, hide_index = True, width = 'stretch')    
    elif model == "DenseNet121":
        st.header('DenseNet121 :')
        st.image('https://pytorch.org/wp-content/uploads/2025/01/densenet1.png',  width = 'stretch')

        st.markdown(
            """
            <div style="text-align: justify;">
            Une autre approche que d’utiliser un modèle "from scratch" est d'utiliser un modèle qui a déjà été entraîné, même sur d'autres tâches de classification. 
            On parle alors de transfer learning. Cela permet généralement d'avoir de meilleurs résultats, ces modèles pré-entraînés possédant de bonnes capacités de généralisation.
            On choisit ici d'utiliser cette approche sur le modèle DenseNet121.
            <br>
            <br>
            La structure du modèle réutilisant DenseNet121 est la suivante :

            <br>
            <br>

            - Couche de conversion grayscale vers RGB
            - Modèle de base DenseNet121
            - Couche de pooling
            - Couche Dropout 30%
            - Couche Dense 256 neurones, activation ReLU
            - Couche Dropout 30%
            - Couche Dense 4 neurones, activation Softmax



            <br>
            <br>

            
            Le modèle est entraîné une première fois avec le modèle de base gelé, sur quelques epochs avec learning rate à 10<sup>-3</sup> puis à nouveau en réduisant le learning rate lors du fine tuning à 10<sup>-4</sup>.
            Tout comme pour le CNN, on utilise une fonction de rappel basée sur un <b>EarlyStopping</b> avec une <b>patience à 15</b> sur le <b>F1-score de validation</b>.

            </div>

            <br>
            <br>

            """
            , unsafe_allow_html=True
        ) 
        df = pd.DataFrame(
            {
            "Classe": ["COVID", "Macro avg"]
            , "Recall": ["0.81","0.88"]
            , "F1-score" : ["0.83","0.89"]
            }
        )
        st.subheader('Résultats obtenus sur ce modèle:')    
        st.dataframe(df, hide_index = True, width = 'stretch')    
    elif model == "EfficientNetB2":
        st.header('EfficientNetB2 :')
        st.image('https://1.bp.blogspot.com/-Cdtb97FtgdA/XO3BHsB7oEI/AAAAAAAAEKE/bmtkonwgs8cmWyI5esVo8wJPnhPLQ5bGQCLcBGAs/s1600/image4.png',  width = 'stretch')

        st.markdown(
            """
            <div style="text-align: justify;">
            Enfin, en se basant sur de la bibliographie existante, le modèle pré-entraîné EfficientNet semble être un excellent candidat concernant la classification d’imagerie médicale pulmonaire par pathologie.
            Le modèle EfficientNetB2 a été choisi pour sa proximité avec la taille des images utilisées (256 x 256 contre 260 x 260 pour le modèle B2) mais des performances supérieures auraient potentiellement pu être obtenues avec un modèle supérieur, typiquement EfficientNetB7. Ce dernier a été entraîné sur des images 600 x 600 et nécessite une quantité de VRAM bien supérieure à ce qui était disponible de notre côté pour vérifier cette hypothèse.
            <br>
            <br>
            La structure du modèle réutilisant EfficientNetB2 est la suivante :

            <br>
            <br>

            - Couche de convulation avec 3 neurones, pas d'activation, kernel size (1,1)
            - Modèle de base EfficientNetB2 (inclut les couches de pooling)
            - Couche Dropout 30%
            - Couche Dense à 4 neurones, activation Softmax

            <br>
            <br>

            
            Le modèle est entraîné une première fois avec le modèle de base gelé, sur quelques epochs avec learning rate à 10<sup>-3</sup> puis à nouveau en réduisant le learning rate lors du fine tuning à 10<sup>-5</sup>.
            Comme pour le CNN, on utilise une fonction de rappel basée sur un <b>EarlyStopping</b> avec une <b>patience à 50</b> sur la <b>perte de validation</b>.

            </div>

            <br>
            <br>

            """
            , unsafe_allow_html=True
        ) 
        df = pd.DataFrame(
            {
            "Classe": ["COVID", "Macro avg"]
            , "Recall": ["0.91","0.93"]
            , "F1-score" : ["0.92","0.93"]
            }
        )
        st.subheader('Résultats obtenus sur ce modèle:')    
        st.dataframe(df, hide_index = True, width = 'stretch')  

    if "selected_button" in st.session_state and st.session_state.selected_button not in options_mod:
        del st.session_state.selected_button



    # def set_model_2(name):
    #     st.session_state.model_choice_2 = name

    # if "model_choice" not in st.session_state:
    #     st.session_state.model_choice_2 = "Random Forest"

    # cols = st.columns(len(options_mod))

    # for i, option_mod in enumerate(options_mod):
    #     with cols[i]:
    #         # Le bouton prend le style "primary" SEULEMENT si son nom correspond à l'option active
    #         #is_selected = (st.session_state.selected_button == option)
    #         is_selected = (st.session_state.model_choice_2 == option_mod)
            
    #         #if st.button(
    #         st.button(
    #             option_mod, 
    #             key=f"btn_2_{i}", 
    #             width = 'stretch',
    #             type="primary" if is_selected else "secondary",
    #             on_click=set_model_2,
    #             args=(option_mod,)
    #         )


        
        
        

if page == pages[4]:
    ### Prédiction

    options = ["CNN", "DenseNet121", "EfficientNetB2"]

    if "selected_button" in st.session_state and st.session_state.selected_button not in options:
        del st.session_state.selected_button

    st.title("Prédiction")

    st.header("Modèle à utiliser:")

    model_dict = {'CNN' : cnn_model
                  , 'DenseNet121' : dense_model
                  , 'EfficientNetB2' : eff_model}

    # if "selected_button" not in st.session_state:
    #     st.session_state.selected_button = "CNN" # Option par défaut
    #     model = model_dict['CNN']
    def set_model(name):
        st.session_state.model_choice = name

    

    if "model_choice" not in st.session_state or st.session_state.model_choice not in options:
        st.session_state.model_choice = "CNN"

    # Affichage des 4 boutons côte à côte
    cols = st.columns(len(options))

    for i, option in enumerate(options):
        with cols[i]:
            # Le bouton prend le style "primary" SEULEMENT si son nom correspond à l'option active
            #is_selected = (st.session_state.selected_button == option)
            is_selected = (st.session_state.model_choice == option)
            
            #if st.button(
            st.button(
                option, 
                key=f"btn_{i}", 
                width = 'stretch',
                type="primary" if is_selected else "secondary",
                on_click=set_model,
                args=(option,)
            )#:
                #st.session_state.selected_button = option
                #st.session_state.model_choice = option

                #st.rerun() # Force la mise à jour immédiate de l'affichage


    #model = model_dict[st.session_state.selected_button]
    model = model_dict[st.session_state.model_choice]

    st.header("Image à prédire:")

    fig_x, ax_x = plt.subplots(figsize=(8, 5))


    img_idx = st.slider("Index de l'image :", min_value = 0, max_value = 15, step = 1,
                        key="img_idx_pred")  # clé fixe, indépendante du modèle choisi)

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

    #if st.session_state.selected_button == "CNN":
    if st.session_state.model_choice == "CNN":
        # fig_mod, class_report = cnn.evaluate_cnn_model(
        #     model,
        #     X_test,
        #     y_test,
        #     img_idx
        # )

        # fig_mod = cnn.explain(model, X_test, y_test, img_idx)
        # st.pyplot(fig_mod)
        if st.session_state.get("cnn_gradcam1_key") != img_idx:
            st.session_state.cnn_fig_mod = cnn.explain(model, X_test, y_test, img_idx)
            st.session_state.cnn_gradcam1_key = img_idx
        
        st.pyplot(st.session_state.cnn_fig_mod)   

        def set_class(name):
            st.session_state.class_choice = name

        if "class_choice" not in st.session_state:
            st.session_state.class_choice = "Normal"
        
        options_explo = ["Normal", "COVID", "Lung Opacity", "Viral Pneumonia"]

        # Affichage des 4 boutons côte à côte
        cols = st.columns(len(options_explo))

        for i, option in enumerate(options_explo):
            with cols[i]:
                # Le bouton prend le style "primary" SEULEMENT si son nom correspond à l'option active
                #is_selected = (st.session_state.selected_button == option)
                is_selected = (st.session_state.class_choice == option)#class_choice
                
                #if st.button(
                st.button(
                    option, 
                    key=f"btn_{i}_D", 
                    width = 'stretch',
                    type="primary" if is_selected else "secondary",
                    on_click=set_class,
                    args=(option,)
                )#:
                    #st.session_state.selected_button = option
                    #st.session_state.class_choice = option
                    #class_name = option
                    #st.rerun() # Force la mise à jour immédiate de l'affichage
        # adapte ce mapping à ton dict_class réel (0=COVID, 1=Lung Opacity, 2=Normal, 3=Viral Pneumonia)
        class_label_map = {"COVID": 0, "Lung Opacity": 1, "Normal": 2, "Viral Pneumonia": 3}
        label = class_label_map[st.session_state.class_choice]

        # clé qui identifie le dernier calcul effectué
        current_key = (img_idx, label)

        if st.session_state.get("cnn_gradcam2_key") != current_key:
            st.session_state.cnn_fig_mod2 = cnn.explain(
                model, X_test, y_test, img_idx, class_idx=label
            )
            st.session_state.cnn_gradcam2_key = current_key

        st.pyplot(st.session_state.cnn_fig_mod2)

        df = pd.read_csv('cnn_final_classification_report.csv',index_col=0)
        st.header('Performances globales du modèle :')

        st.dataframe(df.style.format("{:.2f}"), width = 'stretch')

    #if st.session_state.selected_button == "EfficientNetB2":
    if st.session_state.model_choice == "EfficientNetB2":
        # fig_mod, class_report = eff.evaluate_efficientnet_model(
        #     model,
        #     X_test,
        #     y_test,
        #     img_idx
        # )
        # fig_mod = eff.explain(model, X_test, y_test, img_idx)
        # st.pyplot(fig_mod)
        if st.session_state.get("eff_gradcam1_key") != img_idx:
            st.session_state.eff_fig_mod = eff.explain(model, X_test, y_test, img_idx)
            st.session_state.eff_gradcam1_key = img_idx
        
        st.pyplot(st.session_state.eff_fig_mod)

        def set_class(name):
            st.session_state.class_choice = name

        if "class_choice" not in st.session_state:
            st.session_state.class_choice = "Normal"
        
        options_explo = ["Normal", "COVID", "Lung Opacity", "Viral Pneumonia"]

        # Affichage des 4 boutons côte à côte
        cols = st.columns(len(options_explo))

        for i, option in enumerate(options_explo):
            with cols[i]:
                # Le bouton prend le style "primary" SEULEMENT si son nom correspond à l'option active
                #is_selected = (st.session_state.selected_button == option)
                is_selected = (st.session_state.class_choice == option)#class_choice
                
                #if st.button(
                st.button(
                    option, 
                    key=f"btn_{i}_D", 
                    width = 'stretch',
                    type="primary" if is_selected else "secondary",
                    on_click=set_class,
                    args=(option,)
                )#:
                    #st.session_state.selected_button = option
                    #st.session_state.class_choice = option
                    #class_name = option
                    #st.rerun() # Force la mise à jour immédiate de l'affichage
        # adapte ce mapping à ton dict_class réel (0=COVID, 1=Lung Opacity, 2=Normal, 3=Viral Pneumonia)
        class_label_map = {"COVID": 0, "Lung Opacity": 1, "Normal": 2, "Viral Pneumonia": 3}
        label = class_label_map[st.session_state.class_choice]

        # clé qui identifie le dernier calcul effectué
        current_key = (img_idx, label)

        if st.session_state.get("eff_gradcam2_key") != current_key:
            st.session_state.eff_fig_mod2 = eff.explain(
                model, X_test, y_test, img_idx, class_idx=label
            )
            st.session_state.eff_gradcam2_key = current_key

        st.pyplot(st.session_state.eff_fig_mod2)

        df = pd.read_csv('efficientnetb2_final_classification_report.csv',index_col=0)
        st.header('Performances globales du modèle :')

        st.dataframe(df.style.format("{:.2f}"), width = 'stretch')

    #if st.session_state.selected_button == "DenseNet121":
    if st.session_state.model_choice == "DenseNet121":
        #fig_mod, class_report = gc_densenet.explain(dense_model, X_test, y_test, img_idx)
        #fig_mod = gc_densenet.explain(dense_model, X_test, y_test, img_idx)#,base_model_name="densenet121")
        #st.pyplot(fig_mod)
        if st.session_state.get("dense_gradcam1_key") != img_idx:
            st.session_state.dense_fig_mod = gc_densenet.explain(dense_model, X_test, y_test, img_idx)
            st.session_state.dense_gradcam1_key = img_idx

        st.pyplot(st.session_state.dense_fig_mod)

        def set_class(name):
            st.session_state.class_choice = name

        if "class_choice" not in st.session_state:
            st.session_state.class_choice = "Normal"
        
        options_explo = ["Normal", "COVID", "Lung Opacity", "Viral Pneumonia"]

        # Affichage des 4 boutons côte à côte
        cols = st.columns(len(options_explo))

        for i, option in enumerate(options_explo):
            with cols[i]:
                # Le bouton prend le style "primary" SEULEMENT si son nom correspond à l'option active
                #is_selected = (st.session_state.selected_button == option)
                is_selected = (st.session_state.class_choice == option)#class_choice
                
                #if st.button(
                st.button(
                    option, 
                    key=f"btn_{i}_D", 
                    width = 'stretch',
                    type="primary" if is_selected else "secondary",
                    on_click=set_class,
                    args=(option,)
                )#:
                    #st.session_state.selected_button = option
                    #st.session_state.class_choice = option
                    #class_name = option
                    #st.rerun() # Force la mise à jour immédiate de l'affichage
        # adapte ce mapping à ton dict_class réel (0=COVID, 1=Lung Opacity, 2=Normal, 3=Viral Pneumonia)
        class_label_map = {"COVID": 0, "Lung Opacity": 1, "Normal": 2, "Viral Pneumonia": 3}
        label = class_label_map[st.session_state.class_choice]

        # fig_mod2 = gc_densenet.explain(dense_model, X_test, y_test, img_idx, class_idx=label)
        # st.pyplot(fig_mod2)
        
        # clé qui identifie le dernier calcul effectué
        current_key = (img_idx, label)

        if st.session_state.get("dense_gradcam2_key") != current_key:
            st.session_state.dense_fig_mod2 = gc_densenet.explain(
                dense_model, X_test, y_test, img_idx, class_idx=label
            )
            st.session_state.dense_gradcam2_key = current_key

        st.pyplot(st.session_state.dense_fig_mod2)
        
        df = pd.read_csv('densenet121_final_classification_report.csv',index_col=0)
        st.header('Performances globales du modèle :')
        
        st.dataframe(df.style.format("{:.2f}"), width='stretch')

if page == pages[5] :

    st.title("Conclusion")

    st.header(f"Récapitulatif")

    ### Résumé
    df1 = pd.read_csv('cnn_final_classification_report.csv',index_col=0)
    df2 = pd.read_csv('densenet121_final_classification_report.csv',index_col=0)
    df3 = pd.read_csv('efficientnetb2_final_classification_report.csv',index_col=0)

    df_agg = pd.concat([df1.loc[['COVID', 'macro avg']], df2.loc[['COVID', 'macro avg']], df3.loc[['COVID', 'macro avg']]], keys=['CNN', 'DenseNet121', 'EfficientNetB2'])
    df_agg.index.names = ['Modèle', 'Classe']

    styled_df = df_agg.style.background_gradient(
        cmap="RdYlGn",
        vmin=0.73,  # Valeur min pour le rouge pur
        vmax=0.95,  # Valeur max pour le vert pur
        axis=None,  # Applique le dégradé sur l'ensemble du tableau (ou axis=0 pour par colonne)
    ).format("{:.2f}")

    st.dataframe(styled_df, width = 'stretch', height = 'content')

    st.markdown(
        """
        <div style="text-align: justify;">
        Les deux modèles effectuant du <i>transfer learning</i> ont des performances supérieures au modèle dit <i>baseline</i> et également bien supérieurs aux modèles de machine learning classique.
        On constate qu'EfficientNetB2 performe également mieux que les autres modèles testés. Cela est cohérent avec les résultats déjà obtenus dans la littérature sur des problématiques similaires.
        </div>
        <br>
        <br>
        """
        , unsafe_allow_html=True
    )

    st.header(f"Pistes d'optimisations")

    st.markdown(
        """
        <div style="text-align: justify;">
        Afin d'affiner encore les prédictions, il serait possible de mettre en place un système de vote ou de boosting faisant intervenir les deux modèles les plus performants. De même, on peut supposer qu'un entraînement sans les contraintes techniques rencontrées, notamment sur DenseNet121, permette d'obtenir de meilleurs résultats. On peut également penser que l'entraînement de modèles plus performants, par exemple EfficientNetB7, mais également plus gourmands en ressources, permette encore d'affiner les prédictions.
        </div>
        <br>
        <br>
        """
        , unsafe_allow_html=True
    )

    st.header(f"Difficultés rencontrées")

    st.markdown(
        """
        <div style="text-align: justify;">
        Une des difficultés principales consiste à tenter d’interpréter les résultats. Malgré les GradCAM, qui, s’ils permettent de vérifier si le modèle s’intéresse à des zones cohérentes, il n’est pas aisé de savoir si les zones choisies sont cohérentes d’un point de vue médical. Seules une expertise métier permettrait de réellement valider les zones ciblées. On doit donc se contenter des performances brutes du modèle pour ces analyses.
        <br>
        <br>
        De par la nature du jeu de données (grand nombre d’images de taille modérée), la consommation de la mémoire vive ainsi que de la mémoire vidéo peuvent se retrouver assez vite saturée. Quelques optimisations, notamment sur les types de données utilisées dans les numpy arrays (float32 vs uint8) ont permis de soulager les systèmes. Cela a aussi eu un impact sur les modèles, notamment de transfer learning, qui ont pû être sélectionnés ou sur les stratégies d’entraînement (subsampling pour maintenir un temps de traitement acceptable). Enfin, le modèle DenseNet121 nécessite une étape de normalisation qui nous a obligé à utiliser des float32, ayant un impact direct sur la mémoire utilisée ; EfficientNetB2 n’ayant pas cette contrainte, cela rend son utilisation plus simple.
                
        </div>
        <br>
        <br>
        """
        , unsafe_allow_html=True
    )