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
img = Image.open('img/logo_etools2.png')
st.image(img, width=600)

st.title("PROGRAMME DOCUMENT SIMILARITIES REPORT")

#DATA PREPROCESSING
df = pd.read_csv('data/pd_clusters.csv')
df_3d = pd.read_csv('data/pd_3d_clusters.csv')
df_all_cluster = pd.read_csv('data/pd_all_cluster.csv')
df_matrix = pd.read_csv('data/pd_matrix_similarity.csv').iloc[0:100,0:100] #sample of 100 
df_pd = pd.read_csv('data/pd_data.csv')

data_sun = df_all_cluster.groupby(by=['cluster','country_name','section_name']).nunique()['title'].sort_values(ascending = False).reset_index()

data_bar_country = df_all_cluster.groupby(by=['country_name']).nunique()['title'].reset_index().sort_values(by=['title'],ascending = True)

data_bubble = df.groupby(by = 'cluster').count()['id'].sort_values(ascending = False).reset_index()
data_bubble.columns = ['cluster','#title']



#GLOBAL VARIABLES
max_cluster = df.groupby(by='cluster').count()['title'].reset_index()
MAX_CLUSTER = max_cluster[max_cluster.title==max(max_cluster.title)].values[0][0] 
MAX_CLUSTER_VALUES = max_cluster[max_cluster.title==max(max_cluster.title)].values[0][1] 


data_table_cluster_max = df.query("cluster==@MAX_CLUSTER")[['cluster','title']]

#data comparing all pd's vs duplicates pd's per country
data_all = df_pd.groupby(by='country_name').nunique()['title']
data_duplicates = df_all_cluster.groupby(by='country_name').nunique()['cluster']

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
    data_table_cluster_max = df.query("cluster==@MAX_CLUSTER")[['cluster','title']]
else:
    table = df[['cluster','title']].query("cluster==@add_selectbox")
    #table = df[['cluster','title']].query("cluster==@add_slider")
    df = df.query("cluster==@add_selectbox")
    df_3d = df_3d.query("cluster==@add_selectbox")
    data_table_cluster_max = df.query("cluster==@add_selectbox")[['cluster','title']]

if (len(add_select_countrybox)==0):
    data_sun = df_all_cluster.groupby(by=['cluster','country_name','section_name']).nunique()['title'].sort_values(ascending = False).reset_index()
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
                  width = 500, height = 500,
                  title = "PD's indicators against similar clusters by country & section")

#sunburst # titles by cluser and section
fig_sun_section = px.sunburst(data_sun, 
                  path=['section_name','cluster'], 
                  values='title', color= 'section_name',
                  width = 500, height = 500, color_continuous_scale='blues',
                  title = "PD's indicators against similar clusters by section")

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

#plotly table for max titles in MAS CLUSTER
fig_table_cluster = go.Figure(data=[go.Table(
                        columnwidth = [200, 1000],
                            header=dict(values=list(data_table_cluster_max.columns),  
                                        align='left'),
                            cells=dict(values=[data_table_cluster_max.cluster, data_table_cluster_max.title],        
                                    align='left'))
                        ])
fig_table_cluster.update_layout(
    width=600,
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
st.write("The purpose of this work was to gather the PD indicator titles that are similar to each other and that are being used by different countries in the PRP module from [eTools](https://etools.unicef.org/).")
st.write("To complete this work we proceeded using some Natural Language Processing (NLP) methods ([Sentence Transformer](https://huggingface.co/sentence-transformers)) and obtain semantic information "\
            "from the titles of each PD, so we can find similar sentences not only comparing words, but also its context. "\
                "These methods were only applied to those PD titles available in English corresponding 89% of all registered PD's **unique** title indicators.")   

st.write("Once the similarity matrix is builted, only those PD titles with a similarity between 95% and 99% are filtered for the analysis.")

st.subheader("Let's explore the data through visuals")

st.write("We are going to approach the work obtaining the PD's **unique** titles and then construct our NLP text corpus. Next find a similarity percentage of each sentence"\
        " against all, with this step we can build a similarity matrix where each cell value correspond to a percentage of similarity for all PD's titles. "\
            "")

st.markdown("- Number unique PD titles found on PD-Indicators: **"+str(df.title.nunique())+"**")
st.markdown("- Number of clusters with similarity between 95% and 99%: **"+str(df.cluster.nunique())+"**")


#BLOCK 2
st.write("##")

col1, col2 = st.columns(2)

col1.write("If we analyze the graph below we can identify that cluster "+str(MAX_CLUSTER)+" contains "+str(MAX_CLUSTER_VALUES)+" PD's titles which context are similar.")
col1.plotly_chart(fig_cluster_bubble)



col2.write("Let's show table data related with cluster "+str(MAX_CLUSTER)+":")
col2.plotly_chart(fig_table_cluster)

#BLOCK 3
st.write("At first glance all PD's title looks similar for cluster "+str(MAX_CLUSTER)+", with some differences including especial characters like '#' or words order, "\
        "but in general sentences have almost the same purpose.")
st.write("Now let's explore Vectors of the PD titles through a 3d Graph, this will give us an idea of how the texts are distributed in 3d space. ")
st.write("A sample of matrix similarity using the [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity) method is also presented. "\
            "Similar text are painted with a dark blue tone. The transverse line represents a sentence compared to itself (100% similarity), so it is not relevant for our analysis and is discarded.")

col3, col4 = st.columns(2)

col3.plotly_chart(fig_3d)
col4.plotly_chart(fig_matrix_heatmap)

#BLOCK 4
st.subheader("Merging similar clusters with PD-indicators data")

st.write("With all the information generated, now is the time to combine the data with the information from the [pd-indicators](https://datamart.unicef.io/api/latest/datamart/pd-indicators/) dataset. "\
    "With this process we will be able to find some interesting **Insights** analyzing which countries, sections and other attributes are generating duplicity of information based on "\
        "PD's titles indicators.")


#BLOCK 5

col5, col6 = st.columns(2)

col5.dataframe(df_duplicates.sort_values(by='percentage',ascending=False))

col6.write("**Insight 1: Which country has the highest percentage duplicity against all pd indicators?**")
col6.write("From the left table, data displayed includes the number of unique pd-indicators (title), the number of repeated pd-indicators "\
    "(duplicates) and the ratio between both columns. All rows with country aggregation.")

col6.write("Angola concentrates the highest percentage PD's duplicates with almost 17% compared with all PD's registered"\
            ", followed by Papua New Guinea with 11% and Sudan with 10%. Although Jordan has the largest number of PD's, duplication only represents 31%")

#BLOCK 8
st.write("##")

st.write("**Insight 2: How similar pd-indicators clusters behave against Sections & Countries**")

st.write("Unlike the data obtained for insight 1, we will now approach an analysis only with similar "\
        "pd-indicators, in this way we will be able to understand the classification for duplicated information regarding section type and country.")
col7, col8 = st.columns(2)


col7.write("We observe through a radial graph distribution acrross sections, we can find **Child Protection** as the highest number of similar PD-indicators's, "\
            "followed by **Education** and **WASH**")
col7.plotly_chart(fig_sun_section)

col8.write("Finally ploting and aggregating data by country, Somalia gather the highest number of pd-indicators duplicated. ")
col8.plotly_chart(fig_sun)

st.subheader("Conclusions")

st.write("NLP – Transformers techniques have been used to extract semantic information from the pd-indicators registered in the PRP "\
            "module, this method has allowed us to build a similarity matrix to determine the indicators with a similar objective.")

st.write("With the information analyzed, it is confirmed that there is duplicity in the use of pd-indicators, where the total of 1,347 "\
        "unique pd-indicators registered, 236 these indicators or clusters could be reduced by at least 17.5%.")

st.write("Comparing all the available pd-indicators and the possible duplicates, Angola, Papua New Guinea and Sudan have the highest "\
    "percentage of use of duplicate pd-indicators, however these countries are not the ones with the highest number of pd-indicators. ")

st.write("Finally, similar clusters have been compared, grouping their information by sections and by countries. We observed "\
        "that Child Protection, Education and WASH accumulate the largest number of duplicate pd-indicators by section type, "\
            "while and by country aggregation Somalia, Bangladesh and Iraq accumulate the highest duplicates")

st.write("In the download section you can find the datasets that include the identification of the duplicate pd-indicators.")

#DOWNLOADS 
st.subheader("Downloads")
@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv().encode('utf-8')

st.write("Similar Cluster PD's data across PD-Indicators as CSV")
st.download_button(
     label="Download",
     data=convert_df(df_all_cluster), 
     file_name='all_pd_clusters.csv',
     mime='text/csv',
 )

st.write("Similar Cluster Unique PD-indictaros")
st.download_button(
     label="Download",
     data=convert_df(df),
     file_name='pd_unique_clusters.csv',
     mime='text/csv',
 )


