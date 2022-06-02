import pandas as pd
import matplotlib as plt
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

st.set_page_config(page_title = 'PD Similarities',
                    layout="wide",
                    menu_items={
                                'Get Help': 'https://www.extremelycoolapp.com/help',
                                'Report a bug': "https://www.extremelycoolapp.com/bug",
                                'About': "# This is a header. This is an *extremely* cool app!"
                    }
)

#INSERTING LOGO
img = Image.open('img/logo.png')
st.image(img, caption = 'eTools Report Viz', width=200)

st.title("PROGRAMME DOCUMENT SIMILARITIES REPORT")

#importing clusters dataset
df = pd.read_csv('data/pd_clusters.csv')
df_3d = pd.read_csv('data/pd_3d_clusters.csv')
df_all_cluster = pd.read_csv('data/pd_all_cluster.csv')
df_matrix = pd.read_csv('data/pd_matrix_similarity.csv').iloc[0:100,0:100] #sample of 100 
df_pd = pd.read_csv('data/pd_data.csv')

data_sun = df_all_cluster.groupby(by=['cluster','country_name','section_name','disaggregation_name','location_name']).count()['title'].sort_values(ascending = False).reset_index()
data_bar_country = df_all_cluster.groupby(by=['country_name']).count()['title'].reset_index().sort_values(by=['title'],ascending = True)

data_bubble = df.groupby(by = 'cluster').count()['id'].sort_values(ascending = False).reset_index()
data_bubble.columns = ['cluster','#title']

data_table_cluster_max = df.query("cluster==106")[['cluster','title']]


#data comparing all pd's vs duplicates pd's per country
data_all = df_pd.groupby(by='country_name').count()['title']
data_duplicates = df_all_cluster.groupby(by='country_name').count()['title']

def isDuplicated(value):
    try:
        if data_duplicates[value]>0:
            return True
        else:
            return False
    except:
        return False


duplicates = [data_duplicates[x] if isDuplicated(x) else 0 for x in data_all.index]
percentage_duplicates = [data_duplicates[x]/data_all[x]*100 if isDuplicated(x) else 0 for x in data_all.index]    

df_duplicates = pd.DataFrame(data_all.reset_index())
df_duplicates['duplicates'] = duplicates
df_duplicates['percentage'] = percentage_duplicates




#ADDING SELECTORS TO STREAMLIT DASHBOARD

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.multiselect(
    'Select similar clusters',
    df.cluster.unique()
)

add_select_countrybox = st.sidebar.multiselect(
    'Select Country Name',
    df_all_cluster.country_name.unique()
)

#filtering clusters
if (len(add_selectbox)==0):
    table = df[['cluster','title']]
    df = df
    df_3d = df_3d
    data_table_cluster_max = df.query("cluster==106")[['cluster','title']]
else:
    table = df[['cluster','title']].query("cluster==@add_selectbox")
    #table = df[['cluster','title']].query("cluster==@add_slider")
    df = df.query("cluster==@add_selectbox")
    df_3d = df_3d.query("cluster==@add_selectbox")
    data_table_cluster_max = df.query("cluster==@add_selectbox")[['cluster','title']]

if (len(add_select_countrybox)==0):
    data_sun = df_all_cluster.groupby(by=['cluster','country_name','section_name','disaggregation_name','location_name']).count()['title'].sort_values(ascending = False).reset_index()
else:
    print(add_select_countrybox)
    data_sun = data_sun[data_sun.country_name.isin(add_select_countrybox)]


#FIGURES

fig_table = go.Figure(data=[go.Table(
columnwidth = [200, 1000],
    header=dict(values=list(table.columns),  
                align='left'),
    cells=dict(values=[df.cluster, df.title],        
               align='left'))
])


fig = px.scatter(data_frame = df, x ='x', y='y', 
                 color = 'cluster',
                 hover_data=['title','id'],
                 opacity=1, color_continuous_scale='blues',
                 width = 1000, height = 400)


fig_3d = px.scatter_3d(data_frame = df_3d, x ='x', y='y', z = 'z', 
                 color = 'cluster',
                 hover_data=['id'], hover_name='title',
                 opacity=0.7, color_continuous_scale='blues', template='plotly_dark',
                 width = 600, height = 500,
                 title='3d Vector Semantic most Similar PD titles')


fig_sun = px.sunburst(data_sun, 
                  path=['country_name','section_name','cluster'], 
                  values='title', color= 'country_name',
                  width = 500, height = 500)

#sunburst # titles by cluser and section
fig_sun_section = px.sunburst(data_sun, 
                  path=['section_name','cluster'], 
                  values='title', color= 'section_name',
                  width = 500, height = 500, color_continuous_scale='blues')

#barchart #PD most similar titles by country
fig_bar_country = px.bar(data_bar_country, x='title', y='country_name',
                        text_auto='.2s',
                        height=400, width = 500,
                        color_continuous_scale='blues')

#matrix similarity sample
fig_matrix_heatmap = px.imshow(df_matrix, text_auto = False,
                                title='Matrix Similarity Sample',
                                color_continuous_scale='blues')


#scatter bublles for clusters pd count
fig_cluster_bubble = px.scatter(data_bubble, x="cluster", y="#title",
	         size="#title", color="cluster",
                 hover_name="#title", log_x=True, size_max=20,
                 color_continuous_scale='blues', template='plotly_dark',
                 title = 'Clusters & Number of Similar PDs titles')

#plotly table for max titles in cluster 106
fig_table_cluster = go.Figure(data=[go.Table(
                        columnwidth = [200, 1000],
                            header=dict(values=list(data_table_cluster_max.columns),  
                                        align='left'),
                            cells=dict(values=[data_table_cluster_max.cluster, data_table_cluster_max.title],        
                                    align='left'))
                        ])
fig_table_cluster.update_layout(
    width=650,
    height=400,
    margin=dict(
        l=1,
        r=1,
        b=1,
        t=1
        ),   
    )

#PLOTTING FIGURES TO STREAMLIT DASHBOARD

#BLOCK 1
st.subheader("RESUME")
st.write("The purpose of this work was to gather the PD indicator titles that are similar to each other and that are being used by differente countries in the PRP module.")
st.write("To complete this work we proceeded to use Natural Language Processing (NLP) methods ([Sentence Transformer](https://huggingface.co/sentence-transformers)) and obtain semantic information "\
            "from the titles of each PD, so we can find similar sentences not only comparing words, but also its context. "\
                                "These methods were only applied to those PD titles available in English corresponding 90% of all registered PD's title indicators.")   

st.write("Once the similarity matrix is obtained, only those PD titles with a similarity between 95% and 99% are filtered.")

st.subheader("Let's explore the data through visuals")

st.write("We are going to approach the work obtaining the unique titles of the PD's that construct our NLP text corpus, and then analyze the similarity of each sentence"\
        " against all of them, this step will allow us to build a similarity matrix.")

st.markdown("- Number unique PD titles found on PD-Indicators: **"+str(df.title.nunique())+"**")
st.markdown("- Number of clusters with similarity between 95% and 99%: **"+str(df.cluster.nunique())+"**")


#BLOCK 2
st.write("##")

col1, col2 = st.columns(2)

col1.write("If we analyze the graph below we can identify that cluster 106 contains 23 PD titles that are similar.")
col1.plotly_chart(fig_cluster_bubble)


col2.write("Let's show table data related with cluster 106:")
col2.plotly_chart(fig_table_cluster)


#BLOCK 3
st.write("Now let's explore the vectors of the PD titles through a 3d graph to get an idea of how the texts are distribued in 3d space. "\
        "A sample of matrix similarity using the [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity) method is also presented. Most similar text are painted with a dark blue tone. ")

col3, col4 = st.columns(2)

col3.plotly_chart(fig_3d)
col4.plotly_chart(fig_matrix_heatmap)

#BLOCK 4
st.subheader("Merging similar clusters with PD-indicators data")

st.write("With all the information generated, now is the time to combine the data with the information from the pd-indicators dataset. "\
    "With this process we will be able to analyze which countries, sections and other attributes are generating duplicity of information based on "\
        "PD's titles indicators.")


#BLOCK 5
col5, col6 = st.columns(2)

col5.dataframe(df_duplicates.sort_values(by='percentage',ascending=False))

col6.write("From the table presented on the left, Uganda concentrates the highest percentage PD's duplicates with almost 50% compared with all PD's registered"\
            ", followed by Sudan with 43% and Libya with 32%")

st.plotly_chart(fig_sun)
st.write("Grouping the data by country and by section, we can identify that Jordan concentrates the largest number of duplicate PD's, followed by Libya and Uganda.")




st.subheader("Conclusions")

st.subheader("References")

st.subheader("Downloads")
@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')


st.download_button(
     label="Download Cluster PD's data as CSV",
     data=convert_df(df_all_cluster),
     file_name='all_pd_clusters.csv',
     mime='text/csv',
 )

