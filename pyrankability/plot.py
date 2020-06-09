import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import altair as alt
from pylab import rcParams

from .common import *

alt.data_transformers.disable_max_rows()

# Given something like:
# A = [4, 10, 1, 12, 3, 9, 0, 6, 5, 11, 2, 8, 7]
# B = [5, 4, 10, 1, 7, 6, 12, 3, 9, 0, 11, 2, 8]
def AB_to_P2(A,B):
    P2 = pd.DataFrame(np.array([A,B]))
    return P2

def spider(P2,file=None,fig_format="PNG",width=5,height=10):
    """
    from pyrankability.plot import spider, AB_to_P2

    A = [4, 10, 1, 12, 3, 9, 0, 6, 5, 11, 2, 8, 7]
    B = [5, 4, 10, 1, 7, 6, 12, 3, 9, 0, 11, 2, 8]
    spider(AB_to_P2(A,B))
    """
    rcParams['figure.figsize'] = width, height

    G = nx.Graph()

    pos = {}
    buffer = 0.25
    step = (2-2*buffer)/P2.shape[1]
    labels={}
    y1 = []
    y2 = []
    y = []
    index = []
    for i in range(P2.shape[1]):
        name1 = "A%d:%d"%(i+1,P2.iloc[0,i])
        name2 = "B%d:%d"%(i+1,P2.iloc[1,i])
        G.add_node(name1)
        G.add_node(name2)
        loc = 1-buffer-(i*step)
        pos[name1] = np.array([-1,loc])
        pos[name2] = np.array([1,loc])
        labels[name1] = P2.iloc[0,i]
        labels[name2] = P2.iloc[1,i]
        y1.append(name1)
        y2.append(name2)
        y.append("A")
        y.append("B")
        index.append(name1)
        index.append(name2)
    y=pd.Series(y,index=index)

    for i in range(P2.shape[1]):
        name1 = "A%d:%d"%(i+1,P2.iloc[0,i])
        ix = np.where(P2.iloc[1,:] == P2.iloc[0,i])[0]
        name2 = "B%d:%d"%(ix+1,P2.iloc[0,i])
        G.add_edge(name1, name2)
    edges = G.edges()

    nx.draw_networkx_labels(G,pos=pos,labels=labels)

    color_map = y.map({"A":"blue","B":"red"})
    nx.draw(G, pos, edges=edges, node_color=color_map)
    if file is not None:
        plt.savefig(file, fig_format="PNG")
    plt.show()
    
def show_score_xstar(xstars,indices=None,group_label="Group",fixed_r=None,resolve_scale=False,columns=1,width=300,height=300):
    all_df = pd.DataFrame(columns=["i","j","x",group_label,"ri","rj"])
    score_df = pd.DataFrame(columns=["num_frac_xstar_upper","num_one_xstar_upper","num_zero_xstar_upper"])
    score_df.index.name = group_label
    ordered_xstars = {}
    for key in xstars.keys():
        x = xstars[key].copy()
        if fixed_r is not None and key in fixed_r:
            r = fixed_r[key]
        else:
            r = x.sum(axis=0)
        order = np.argsort(r)
        xstar = x.copy().iloc[order,:].iloc[:,order]
        xstar.loc[:,:] = threshold_x(xstar.values)
        if indices is not None:
            x = x.iloc[indices[key],:].iloc[:,indices[key]]
        ordered_xstars[key] = xstar
        inxs = np.triu_indices(len(xstar),k=1)
        xstar_upper = xstar.values[inxs[0],inxs[1]]
        nfrac_upper = sum((xstar_upper > 0) & (xstar_upper < 1))
        none_upper = sum(xstar_upper == 1)
        nzero_upper = sum(xstar_upper == 0)
        
        score_df = score_df.append(pd.Series([nfrac_upper,none_upper,nzero_upper],index=score_df.columns,name=key))
        #rixs = np.argsort(r)
        #x = x.iloc[:,rixs].iloc[rixs,:]#np.ix_(rixs,rixs)]
        df = (1-x).stack().reset_index()
        df.columns=["i","j","x"]

        df["ri"] = list(r.loc[df["i"]])
        df["rj"] = list(r.loc[df["j"]])
        df[group_label] = key
        all_df = all_df.append(df)

    #all_df = all_df.loc[(all_df.x != 0) & (all_df.x != 1)]
    g = alt.Chart(all_df,width=width).mark_square().encode(
        x=alt.X(
            'i:N',
            axis=alt.Axis(labelOverlap=False),
            title="r",
            sort=alt.EncodingSortField(field="ri",order="ascending") # The order to sort in
        ),
        y=alt.Y(
            'j:N',
            axis=alt.Axis(labelOverlap=False),
            title="r",
            sort=alt.EncodingSortField(field="rj",order="ascending") # The order to sort in
        ),
        color=alt.Color("x",scale=alt.Scale(scheme='greys'))
    ).properties(
        width=width,
        height=height
    ).facet(
        facet=alt.Column("%s:N"%group_label, title=None),
        columns=columns
    )
    if resolve_scale:
        g = g.resolve_scale(x='independent',y='independent')
    return g,score_df,ordered_xstars

def show_score_xstar2(xstars,indices=None,group_label="Group",fixed_r=None,resolve_scale=False,columns=1,width=300,height=300):
    all_df = pd.DataFrame(columns=["i","j","x",group_label,"ri","rj"])
    score_df = pd.DataFrame(columns=["num_frac_xstar_upper","num_one_xstar_upper","num_zero_xstar_upper"])
    score_df.index.name = group_label
    ordered_xstars = {}
    for key in xstars.keys():
        x = xstars[key].copy()
        if fixed_r is not None and key in fixed_r:
            r = fixed_r[key]
        else:
            r = x.sum(axis=0)
        order = np.argsort(r)
        xstar = x.copy().iloc[order,:].iloc[:,order]
        xstar.loc[:,:] = threshold_x(xstar.values)
        if indices is not None:
            x = x.iloc[indices[key],:].iloc[:,indices[key]]
        # For coloring purposes
        x.loc[:,:] = threshold_x(x.values)
        ordered_xstars[key] = xstar
        inxs = np.triu_indices(len(xstar),k=1)
        xstar_upper = xstar.values[inxs[0],inxs[1]]
        nfrac_upper = sum((xstar_upper > 0) & (xstar_upper < 1))
        none_upper = sum(xstar_upper == 1)
        nzero_upper = sum(xstar_upper == 0)
        score_df = score_df.append(pd.Series([nfrac_upper,none_upper,nzero_upper],index=score_df.columns,name=key))
        #rixs = np.argsort(r)
        #x = x.iloc[:,rixs].iloc[rixs,:]#np.ix_(rixs,rixs)]
        df = x.stack().reset_index()
        df.columns=["i","j","x"]

        df["ri"] = list(r.loc[df["i"]])
        df["rj"] = list(r.loc[df["j"]])
        df.loc[:,"c"] = "white"
        df.loc[(df["x"] > 0) & (df["x"] < 1) & (df["ri"] < df["rj"]),"c"] = "green"
        df.loc[(df["x"] > 0) & (df["x"] < 1) & (df["ri"] > df["rj"]),"c"] = "red"
        df.loc[df["i"] == df["j"],"c"] = "black" 
        df[group_label] = key
        all_df = all_df.append(df)

    #all_df = all_df.loc[(all_df.x != 0) & (all_df.x != 1)]
    g = alt.Chart(all_df,width=width).mark_square().encode(
        x=alt.X(
            'i:N',
            axis=alt.Axis(labelOverlap=False),
            title="r",
            sort=alt.EncodingSortField(field="ri",order="ascending") # The order to sort in
        ),
        y=alt.Y(
            'j:N',
            axis=alt.Axis(labelOverlap=False),
            title="r",
            sort=alt.EncodingSortField(field="rj",order="ascending") # The order to sort in
        ),
        color=alt.Color("c",scale=None)#alt.Scale(scheme='greys'))
    ).properties(
        width=width,
        height=height
    ).facet(
        facet=alt.Column("%s:N"%group_label, title=None),
        columns=columns
    )
    if resolve_scale:
        g = g.resolve_scale(x='independent',y='independent')
    return g,score_df,ordered_xstars