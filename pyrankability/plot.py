import matplotlib.pyplot as plt
import networkx as nx
import nx_altair as nxa
import numpy as np
import pandas as pd
import math
import altair as alt
from pylab import rcParams

from .common import *

alt.data_transformers.disable_max_rows()

from networkx.drawing.nx_agraph import graphviz_layout, to_agraph
import pygraphviz as pgv

from IPython.display import Image

def draw(A):
    return Image(A.draw(format='png', prog='dot'))

def D_as_graph(D,file=None):
    G = nx.DiGraph()
    for i in D.index:
        for j in D.columns:
            if D.loc[i,j] != 0:
                G.add_edge(i,j,width=D.loc[i,j],label=D.loc[i,j])
                
    A = to_agraph(G)
    A.layout('dot')
    if file is not None:
        A.draw(file)
    return draw(A)

# Given something like:
# A = [4, 10, 1, 12, 3, 9, 0, 6, 5, 11, 2, 8, 7]
# B = [5, 4, 10, 1, 7, 6, 12, 3, 9, 0, 11, 2, 8]
def AB_to_P2(A,B):
    P2 = pd.DataFrame(np.array([A,B]))
    return P2

def spider(perm1,perm2,font_size=14):
    assert len(perm1) == len(perm2)
    assert type(perm1) == pd.Series
    assert type(perm2) == pd.Series
    assert perm1.name != perm2.name
    
    G = nx.Graph()

    node_pos = {}
    label_pos = {}
    buffer = 0.25
    step = (2-2*buffer)/len(perm1)
    y1 = []
    y2 = []
    y = []
    index = [] 
    max_length_label = perm1.index.map(lambda x: len(x)).max()
    relative_label_position = max(math.log(max_length_label, 7), 1.2)
    for i in range(len(perm1)):
        name1 = f"{perm1.name}:{perm1.iloc[i]}"
        name2 = f"{perm2.name}:{perm2.iloc[i]}"
        G.add_node(name1, label=perm1.index[i])
        G.add_node(name2, label=perm2.index[i])
        loc = 1-buffer-(i*step)
        node_pos[name1] = np.array([-1,loc])
        node_pos[name2] = np.array([1,loc])
        label_pos[name1] = np.array([-1 * relative_label_position,loc])
        label_pos[name2] = np.array([relative_label_position,loc])
        y1.append(name1)
        y2.append(name2)
        y.append("A")
        y.append("B")
        index.append(name1) 
        index.append(name2)
    y=pd.Series(y,index=index)

    for i in range(len(perm1)):
        name1 = f"{perm1.name}:{perm1.iloc[i]}"
        ix = np.where(perm1.iloc[i] == perm2)[0][0]
        name2 = f"{perm2.name}:{perm2.iloc[ix]}"
        G.add_edge(name1, name2)

    # Creates the spider plot. The calling function will need to handle resizing.
    viz = nxa.draw_altair.draw_networkx(G, pos=node_pos, node_color='white')
    viz = alt.layer(viz, nxa.draw_altair.draw_networkx_labels(G, pos=label_pos, font_size=font_size, node_label='label'))
    return viz

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
        
    g.configure_title(
        fontSize=12,
        font='Times',
        orient='bottom'
    )
    return g,score_df,ordered_xstars  

def show_single_xstar(x,indices=None,fixed_r=None,
                      width=400,height=400,
                      labelFontSize=10,titleFontSize=10,prepare_url_func=None,red_green=True):
    ordered_xstars = {}
    if fixed_r is not None:
        r = fixed_r
    else:
        r = x.sum(axis=1)
    order = np.argsort(r)
    xstar = x.copy().iloc[order,:].iloc[:,order]
    xstar.loc[:,:] = threshold_x(xstar.values)
    if indices is not None:
        x = x.iloc[indices[key],:].iloc[:,indices[key]]
    # For coloring purposes
    x.loc[:,:] = threshold_x(x.values)
    ordered_xstar = xstar
    inxs = np.triu_indices(len(xstar),k=1)
    xstar_upper = xstar.values[inxs]
    nfrac_upper = sum((xstar_upper > 0) & (xstar_upper < 1))
    none_upper = sum(xstar_upper == 1)
    nzero_upper = sum(xstar_upper == 0)
    score_series = pd.Series([nfrac_upper,none_upper,nzero_upper],
                             index=["num_frac_xstar_upper","num_one_xstar_upper","num_zero_xstar_upper"])
    df = x.stack().reset_index()
    df.columns=["i","j","x"]

    df["ri"] = list(r.loc[df["i"]])
    df["rj"] = list(r.loc[df["j"]])
    if red_green:
        df.loc[:,"c"] = "white"
        df.loc[(df["x"] > 0) & (df["x"] < 1) & (df["ri"] < df["rj"]),"c"] = "green"
        df.loc[(df["x"] > 0) & (df["x"] < 1) & (df["ri"] > df["rj"]),"c"] = "red"
        df.loc[df["i"] == df["j"],"c"] = "black" 
        color = alt.Color("c:N",scale=None)
    else:
        df.loc[:,"c"] = 0
        df.loc[(df["x"] > 0) & (df["x"] < 1) & (df["ri"] < df["rj"]),"c"] = 1-(df.loc[(df["x"] > 0) & (df["x"] < 1) & (df["ri"] < df["rj"]),"x"])
        df.loc[(df["x"] > 0) & (df["x"] < 1) & (df["ri"] > df["rj"]),"c"] = (df.loc[(df["x"] > 0) & (df["x"] < 1) & (df["ri"] > df["rj"]),"x"])
        df.loc[df["i"] == df["j"],"c"] = 1
        scale=alt.Scale(domain=[0, 0.0001, 1], range=["white", "white", "grey"])
        #color = alt.Color("c",scale=alt.Scale(scheme="greys"))        
        color = alt.Color("c",scale=scale)

    if prepare_url_func is not None:
        df_url = prepare_url_func(df)
    else:
        df_url = df
    g = alt.Chart(df_url,width=width).mark_square().encode(
        x=alt.X(
            'i:N',
            axis=alt.Axis(labelOverlap=False,labelFontSize=8),
            title="r",
            sort=alt.EncodingSortField(field="ri",order="ascending") # The order to sort in
        ),
        y=alt.Y(
            'j:N',
            axis=alt.Axis(labelOverlap=False,labelFontSize=8),
            title="r",
            sort=alt.EncodingSortField(field="rj",order="ascending") # The order to sort in
        ),
        color=color #alt.Color("c:N",scale=None)#alt.Scale(scheme='greys'))
    ).properties(
        width=width,
        height=height
    ).configure_axis(
        labelFontSize=labelFontSize,
        titleFontSize=titleFontSize
    )
    
    return g,score_series,ordered_xstar

def show_score_xstar2(xstars,indices=None,group_label="Group",fixed_r=None,resolve_scale=False,columns=1,width=300,height=300,labelFontSize=12):
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
        xstar_upper = xstar.values[inxs]
        #import pdb; pdb.set_trace()
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
            axis=alt.Axis(labelOverlap=False,labelFontSize=8),
            title="r",
            sort=alt.EncodingSortField(field="ri",order="ascending") # The order to sort in
        ),
        y=alt.Y(
            'j:N',
            axis=alt.Axis(labelOverlap=False,labelFontSize=8),
            title="r",
            sort=alt.EncodingSortField(field="rj",order="ascending") # The order to sort in
        ),
        color=alt.Color("c",scale=None)#alt.Scale(scheme='greys'))
    ).properties(
        width=width,
        height=height
    ).facet(
        facet=alt.Column(title=None,field=alt.Field(group_label),type='nominal',header=alt.Header(labelFontSize=labelFontSize,labelOrient='bottom')),
        #alt.Column("%s:N"%group_label, title=,header=alt.Header(labelBaseline="bottom")),
        columns=columns
    ).configure_axis(
        labelFontSize=10,
        titleFontSize=10
    )
    
    #g= g.configure_title(
    #    fontSize=12,
    #    font='Times',
    #    titleAnchor='bottom'
    #)
    
    if resolve_scale:
        g = g.resolve_scale(x='independent',y='independent')
    return g,score_df,ordered_xstars

def show_hillside(V,P0):
    perm=pd.Series(P0,index=V.columns)
    r=perm.argsort()
    #V_G=V.iloc[perm,:].iloc[:,perm]

    #x = pd.DataFrame(details['x'],index=V.index,columns=V.columns).iloc[perm,:].iloc[:,perm]
    #r = x.sum(axis=1)

    df=V.T.stack().to_frame().reset_index()
    df.columns=["team_i_name","team_k_name","v"]
    df["ri"] = list(-r.loc[df["team_i_name"]])
    df["rk"] = list(r.loc[df["team_k_name"]])

    g=alt.Chart(df).mark_circle().encode(
        x=alt.X(
            'team_i_name:N',
            axis=alt.Axis(labelOverlap=False),
            title="r",
            sort=alt.SortField(field="ri",order="descending") # The order to sort in
        ),
        y=alt.Y(
            'team_k_name:N',
            axis=alt.Axis(labelOverlap=False),
            title="r",
            sort=alt.SortField(field="rk",order="ascending") # The order to sort in
        ),
        size='v:Q'
    )
    return g
