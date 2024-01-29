from caveclient import CAVEclient
import numpy as np 
import time
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

def check_bounds(pt, min_bounds, max_bounds):
    within = True
    if pt[0] < min_bounds[0] or pt[0] > max_bounds[0]:
        within = False
    if pt[1] < min_bounds[1] or pt[1] > max_bounds[1]:
        within = False
    if pt[2] < min_bounds[2] or pt[2] > max_bounds[2]:
        within = False
    
    return within

def get_inbound_l2ids(l2ids,l2points, soma_pt, min_dist, max_dist):
    inbounds = []
    for i in l2ids:
        pt_dict = l2points[str(i)]
        pt = pt_dict['rep_coord_nm']
        
        dist = np.linalg.norm(soma_pt-pt)
        if dist > min_dist and dist <= max_dist:
            inbounds.append(True)
        else:
            inbounds.append(False)
        
    return l2ids[inbounds]

def get_syn_count(segids, client):
    counts = []
    for i in segids:
        syn_df = client.materialize.synapse_query(post_ids = i, synapse_table = 'synapses_nov2022')
        c = syn_df.shape[0]
        counts.append(c)
    return counts


def get_lvl2_size_metrics(segid, 
                          radius=0, 
                          pt=[], 
                          within_cutout=False,
                          dataset_name = 'fanc_production_mar2021'):
    dataset = dataset_name
    client = CAVEclient(dataset)
    
    lvl2ids = client.chunkedgraph.get_leaves(segid, stop_layer=2)
    l2points = client.l2cache.get_l2data(lvl2ids,['rep_coord_nm'])
    if radius > 0 and within_cutout== True:
        min_dist = 0
        max_dist = radius
        print('Getting soma l2ids')
        lvl2ids = get_inbound_l2ids(lvl2ids,l2points, pt, min_dist, max_dist)
    elif radius > 0 and within_cutout==False:
        min_dist = radius
        max_dist = float('inf')
        print('Getting radial l2ids')
        lvl2ids = get_inbound_l2ids(lvl2ids,l2points, pt, min_dist, max_dist)
    else:
        lvl2ids = client.chunkedgraph.get_leaves(segid, stop_layer=2)
        
    l2attrs = client.l2cache.get_l2data(lvl2ids,['size_nm3'])
    
    v = []
    for i in l2attrs.values():
        if len(i) >0:
            j = i['size_nm3']
            v.append(j)
    volumes = np.array(v)

    tot_vol_um3 = np.sum(volumes)/(1000*1000*1000)
    
    l2attrs_area = client.l2cache.get_l2data(lvl2ids,['area_nm2'])
    
    v = []
    for i in l2attrs_area.values():
        if len(i) >0:
            j = i['area_nm2']
            v.append(j)
    areas = np.array(v)

    tot_area_um2 = np.sum(areas)/(1000*1000)
    
    return tot_vol_um3, tot_area_um2

def add_size_columns(df,cutout_radius = 5000,
                     id_column = 'pt_root_id',
                     pt_column = 'pt_position',
                     area_point = 'soma',
                     within_cutout=True):
    
    segids = df[id_column].tolist()

    vols = []
    areas = []
    cutout_radius = 5000

    for ix, i in enumerate(segids):
        subdf = df[df[id_column]==i]
        pt = subdf[pt_column].tolist()[0]
        pt_nm = pt * np.array([4.3,4.3,45])
        print(i)
        v, a = get_lvl2_size_metrics(i, radius = cutout_radius, pt = pt_nm, within_cutout=within_cutout)
        print(v)
        vols.append(v)
        areas.append(a)
        # if ix%80 == 0:
        #     time.sleep(30)


    df['%s_volume_um3'%area_point] = vols
    df['%s_area_um2'%area_point] = areas

    return df

def add_in_out_degree(df,client,
                      id_column='pt_root_id',
                      synapse_table = 'synapses_nov2022',
                      add_within_connectivity=False):
    segids = df[id_column].tolist() 
    indegree, outdegree = [], []
    for ix,i in enumerate(segids):
        ind = client.materialize.synapse_query(post_ids = [i], synapse_table =synapse_table )
        outd = client.materialize.synapse_query(pre_ids = [i], synapse_table =synapse_table )
        indegree.append(ind.shape[0])
        outdegree.append(outd.shape[0])

        if ix%100 == 0:
            time.sleep(60)

    df['out_degree'] = outdegree
    df['in_degree'] = indegree

    if add_within_connectivity==True:
        synd = client.materialize.synapse_query(post_ids = segids, pre_ids = segids, synapse_table =synapse_table )
        indegree, outdegree = [], []
        for i in segids:
            ind = synd.query('post_pt_root_id==@i').shape[0]
            outd = synd.query('pre_pt_root_id==@i').shape[0]
            indegree.append(ind)
            outdegree.append(outd)

        df['within_group_out_degree'] = outdegree
        df['within_group_in_degree'] = indegree
    
    return df
  

def plot_dendrogram(sim_mat, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None).fit(sim_mat)
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    dend_dict = dendrogram(linkage_matrix, **kwargs)
    
    # sorted order of indices found through clustering
    clustered_order = dend_dict['ivl']
    new_order = []
    for c in clustered_order:
        if '(' in c:
            c = c.split('(')[1]
            c = c.split(')')[0]
        new_order.append(int(c))
    
    return clustered_order, new_order

def get_connectivity_matrix(df,cells, 
                            pre_column = 'pre_pt_root_id',
                            post_column='post_pt_root_id',
                            with_similarity_matrix = False):

    # pre_cells = df[pre_column].unique().tolist()
    # post_cells = df[post_column].unique().tolist()
    num_post = len(cells)
    num_pre = len(cells)
    conn_mat = np.zeros((num_pre, num_post))

    for ix, i in enumerate(cells):
        
        subset = df[df[pre_column]==i]
        for ixc, common in enumerate(cells):
            
            if common in subset[post_column].values:
                
                conn_mat[ix, ixc] = subset[subset[post_column]==common]['count'].tolist()[0]

    if with_similarity_matrix == True:
        sim_mat = cosine_similarity(conn_mat)
        return conn_mat, sim_mat
    else:
        return conn_mat 

def get_edge_table(client, pre_cells, post_cells, synapse_table = 'synapses_nov2022'):
    
    synapse_df = client.materialize.synapse_query(pre_ids = pre_cells, post_ids=post_cells, synapse_table = synapse_table)
    edge_df = synapse_df.groupby(['pre_pt_root_id', 'post_pt_root_id']).size().sort_values(ascending=False).reset_index(name='count')

    return edge_df,synapse_df

