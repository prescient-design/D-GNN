#This script makes the database given PDB files and their corrosponding classes/labels/values

import logging
import numpy
import yaml

from packman import molecule
from scipy.spatial import Delaunay

def Circumsphere(Tetrahydron):
    """Get the Circumsphere of the set of four points.
    
    Given any four three dimentional points, this function calculates the features of the circumsphere having the given four points on it's surface.
    
    Args:
        Tetrahydron ([packman.molecule.Atom] or [[X,Y,Z]]): Either packman.molecule.Atom objects or 3D corrdinates. 
    
    Returns:
        [Centre, Radius] (float): The 3D coordinates of the geometrical center of the given four points, Radius of the circumsphere made up of given four points in that order.
    """
    alpha_mat, gamma_mat, Dx_mat, Dy_mat, Dz_mat=[], [], [], [], []
    for i in Tetrahydron:
        temp_coords = i
        alpha_mat.append( [temp_coords[0],temp_coords[1],temp_coords[2],1] )
        gamma_mat.append( [temp_coords[0]**2+temp_coords[1]**2+temp_coords[2]**2,temp_coords[0],temp_coords[1],temp_coords[2]] )
        Dx_mat.append( [temp_coords[0]**2+temp_coords[1]**2+temp_coords[2]**2,temp_coords[1],temp_coords[2],1] )
        Dy_mat.append( [temp_coords[0]**2+temp_coords[1]**2+temp_coords[2]**2,temp_coords[0],temp_coords[2],1] )
        Dz_mat.append( [temp_coords[0]**2+temp_coords[1]**2+temp_coords[2]**2,temp_coords[0],temp_coords[1],1] )
    alpha = numpy.linalg.det(alpha_mat)
    gamma = numpy.linalg.det(gamma_mat)
    Dx = (+numpy.linalg.det(Dx_mat))
    Dy = (-numpy.linalg.det(Dy_mat))
    Dz = (+numpy.linalg.det(Dz_mat))
    Centre = numpy.array([Dx/2*alpha,Dy/2*alpha,Dz/2*alpha])
    Radius = numpy.sqrt(Dx**2 + Dy**2 + Dz**2 - 4*alpha*gamma)/(2*numpy.absolute(alpha))
    return Centre, Radius

#Constants
kidera={
'ALA':[-1.56,-1.67,-0.97,-0.27,-0.93,-0.78,-0.2,-0.08,0.21,-0.48],
'ARG':[0.22,1.27,1.37,1.87,-1.7,0.46,0.92,-0.39,0.23,0.93],
'ASN':[1.14,-0.07,-0.12,0.81,0.18,0.37,-0.09,1.23,1.1,-1.73],
'ASP':[0.58,-0.22,-1.58,0.81,-0.92,0.15,-1.52,0.47,0.76,0.7],
'CYS':[0.12,-0.89,0.45,-1.05,-0.71,2.41,1.52,-0.69,1.13,1.1],
'GLN':[-0.47,0.24,0.07,1.1,1.1,0.59,0.84,-0.71,-0.03,-2.33],
'GLU':[-1.45,0.19,-1.61,1.17,-1.31,0.4,0.04,0.38,-0.35,-0.12],
'GLY':[1.46,-1.96,-0.23,-0.16,0.1,-0.11,1.32,2.36,-1.66,0.46],
'HIS':[-0.41,0.52,-0.28,0.28,1.61,1.01,-1.85,0.47,1.13,1.63],
'ILE':[-0.73,-0.16,1.79,-0.77,-0.54,0.03,-0.83,0.51,0.66,-1.78],
'LEU':[-1.04,0,-0.24,-1.1,-0.55,-2.05,0.96,-0.76,0.45,0.93],
'LYS':[-0.34,0.82,-0.23,1.7,1.54,-1.62,1.15,-0.08,-0.48,0.6],
'MET':[-1.4,0.18,-0.42,-0.73,2,1.52,0.26,0.11,-1.27,0.27],
'PHE':[-0.21,0.98,-0.36,-1.43,0.22,-0.81,0.67,1.1,1.71,-0.44],
'PRO':[2.06,-0.33,-1.15,-0.75,0.88,-0.45,0.3,-2.3,0.74,-0.28],
'SER':[0.81,-1.08,0.16,0.42,-0.21,-0.43,-1.89,-1.15,-0.97,-0.23],
'THR':[0.26,-0.7,1.21,0.63,-0.1,0.21,0.24,-1.15,-0.56,0.19],
'TRP':[0.3,2.1,-0.72,-1.57,-1.16,0.57,-0.48,-0.4,-2.3,-0.6],
'TYR':[1.38,1.48,0.8,-0.56,0,-0.68,-0.31,1.03,-0.05,0.53],
'VAL':[-0.74,-0.71,2.04,-0.4,0.5,-0.81,-1.07,0.06,-0.46,0.65],
'ASX':[0.86,-0.145,-0.85,0.81,-0.37,0.26,-0.805,0.85,0.93,-0.515],
'GLX':[-0.96,0.215,-0.77,1.135,-0.105,0.495,0.44,-0.165,-0.19,-1.225],
'XAA':[0,0,0,0,0,0,0,0,0,0],
'XLE':[-0.885,-0.08,0.775,-0.935,-0.545,-1.01,0.065,-0.125,0.555,-0.425]
}

OHE = {
'ALA':[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
'ARG':[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
'ASN':[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
'ASP':[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
'CYS':[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
'GLN':[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
'GLU':[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
'GLY':[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
'HIS':[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
'ILE':[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
'LEU':[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
'LYS':[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
'MET':[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
'PHE':[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
'PRO':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
'SER':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
'THR':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
'TRP':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
'TYR':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
'VAL':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
'ASX':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
'GLX':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
'XAA':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
'XLE':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]
}


# Input config file
with open('config.yaml', 'r') as file:
    yaml_input = yaml.safe_load(file)
if(yaml_input['node_features']=='kidera'):
    nf = kidera
elif(yaml_input['node_features']=='OHE'):
    nf = OHE
else:
    print('Please provide the right node_feature paramemter. Read the config.yaml for details.')
    exit()

def single_entry_process(args):
    '''
    '''
    splitted_line, adj_thr, location = args

    adj_type, thr = adj_thr.split('_')
    thr = float(thr)
    
    try:
        mol = molecule.load_structure(location+splitted_line[0])
    except:
        print(splitted_line[0],' could not be loaded. (Hint: Check structure_location parameter in the yaml file)')
        return
    
    # Extract chains according to the data
    try:
        model = int(splitted_line[1])
        H_chain = splitted_line[2]
        L_chain = splitted_line[3]

        #'y' is made float here regardless of if it's a classification or regression problem
        y = float(splitted_line[5])
    except Exception as e:
        print(splitted_line, str(e))
        return
    
    if(H_chain=='NA' or L_chain=='NA'):
        logging.error('Either Heavy or L chain is NA for '+splitted_line[0])
        return
    
    # Extract residues from those atoms
    try:
        residues = [i for i in mol[model][H_chain].get_residues() if i.get_calpha()!=None] + [i for i in mol[model][L_chain].get_residues() if i.get_calpha()!=None]
    except Exception as e:
        print(splitted_line, str(e))
        return

    # If Antigen residues are present
    if(splitted_line[4]!='NA'): 
        AgChains = [i.strip() for i in splitted_line[4].split('|')]
        
        if(H_chain in AgChains or L_chain in AgChains):
            logging.error('Either Heavy or L chain is same as AgChain for '+splitted_line[0])
            return
        
        # Add the antigen residues
        antigen_residues = []
        try:
            for Ag in AgChains:
                antigen_residues = antigen_residues + [i for i in mol[model][Ag].get_residues() if i.get_calpha()!=None]
        except:
            logging.error('Antigen chain missing in '+splitted_line[0])
            return
        
        # For all kinds of purposes
        all_residues = residues + antigen_residues

        # Index consistent; !!One extra zero in OHE for the interface or not
        node_features = []
        for i in all_residues:
            which_chain = None
            if(i.get_parent().get_id() == H_chain):
                which_chain = [1,0,0,0]
            elif(i.get_parent().get_id() == L_chain):
                which_chain = [0,1,0,0]
            elif(i.get_parent().get_id() in AgChains):
                which_chain = [0,0,1,0]
            else:
                print(H_chain,L_chain,AgChains, i.get_parent().get_id())

            try:
                node_features.append( nf[i.get_name()] + which_chain )
            except:
                #Push alanine if residue is not one of the known
                node_features.append( nf['ALA'] + which_chain )
        node_features = numpy.array( node_features )

    else:
        # Index consistent
        node_features = []
        for i in residues:
            try:
                node_features.append( nf[i.get_name()] )
            except:
                #Push alanine if residue is not one of the known
                node_features.append( nf['ALA'] )
        node_features = numpy.array( node_features )

    # Delaunay
    if(adj_type=='DT'):
        # Ab only
        if(splitted_line[4] == 'NA'):
            #ADJ tensor for all!
            adj = numpy.identity(len(residues))

            #Calpha locations
            calpha_locations = [i.get_calpha().get_location() for i in residues if i.get_calpha() != None]

            DT = Delaunay( calpha_locations )

            for i in DT.simplices:
                center, radius = Circumsphere([ calpha_locations[i[0]], calpha_locations[i[1]], calpha_locations[i[2]], calpha_locations[i[3]] ])

                if(radius <= thr):

                    adj[i[0]][i[1]] = 1
                    adj[i[1]][i[0]] = 1

                    adj[i[0]][i[2]] = 1
                    adj[i[2]][i[0]] = 1

                    adj[i[0]][i[3]] = 1
                    adj[i[3]][i[0]] = 1

                    adj[i[1]][i[2]] = 1
                    adj[i[2]][i[1]] = 1

                    adj[i[1]][i[3]] = 1
                    adj[i[3]][i[1]] = 1

                    adj[i[2]][i[3]] = 1
                    adj[i[3]][i[2]] = 1
            
        #Ab and Ag
        elif(splitted_line[4] != 'NA'):

            calpha_locations = [i.get_calpha().get_location() for i in all_residues if i.get_calpha() != None]

            #ADJ tensor for all!
            adj = numpy.identity(len(all_residues))

            DT = Delaunay( calpha_locations )

            #Go over all the tessellations
            for i in DT.simplices:
                center, radius = Circumsphere([ calpha_locations[i[0]], calpha_locations[i[1]], calpha_locations[i[2]], calpha_locations[i[3]] ])
                all_chains = list(set([ all_residues[j].get_parent().get_id() for j in i]))

                if(radius <= thr):

                    adj[i[0]][i[1]] = 1
                    adj[i[1]][i[0]] = 1

                    adj[i[0]][i[2]] = 1
                    adj[i[2]][i[0]] = 1

                    adj[i[0]][i[3]] = 1
                    adj[i[3]][i[0]] = 1

                    adj[i[1]][i[2]] = 1
                    adj[i[2]][i[1]] = 1

                    adj[i[1]][i[3]] = 1
                    adj[i[3]][i[1]] = 1

                    adj[i[2]][i[3]] = 1
                    adj[i[3]][i[2]] = 1

                    # If a tessellation contains heavy or light AND antigen chain (a.k.a is in an interface) then the last feature is '1'
                    if( len(list(set(AgChains).intersection(all_chains))) >0 and len(list(set([H_chain,L_chain]).intersection(all_chains))) >0 ):
                        node_features[i[0]][len(node_features[0])-1] = 1
                        node_features[i[1]][len(node_features[0])-1] = 1
                        node_features[i[2]][len(node_features[0])-1] = 1
                        node_features[i[3]][len(node_features[0])-1] = 1
                    else:
                        None
            
    # KNN
    elif(adj_type=='KNN'):
        from sklearn.neighbors import kneighbors_graph

        # Ab only
        if(splitted_line[4] == 'NA'):
            # +1 is because include_self makes self count as neighbour so KNN 9 becomes KNN 8
            G = kneighbors_graph( [i.get_calpha().get_location() for i in residues], n_neighbors= int(thr)+1, include_self= True )
            adj = numpy.array(G.toarray())

        # Ab - Ag
        else:
            G = kneighbors_graph( [i.get_calpha().get_location() for i in all_residues], n_neighbors= int(thr)+1, include_self= True )
            adj = numpy.array(G.toarray())

    # Cutoff
    elif(adj_type=='Cutoff'):
        
        #Ab only
        if(splitted_line[4] == 'NA'):
            adj = numpy.identity(len(residues))
            calpha_locations = [i.get_calpha().get_location() for i in residues if i.get_calpha() != None]

            for i in range(0,len(calpha_locations)):
                for j in range(i+1,len(calpha_locations)):
                    if(residues[i].get_calpha().calculate_distance(residues[j].get_calpha()) <= thr ):
                        adj[i][j] = 1
                        adj[j][i] = 1
        
        # Ab - Ag
        else:
            adj = numpy.identity(len(all_residues))
            calpha_locations = [i.get_calpha().get_location() for i in all_residues if i.get_calpha() != None]

            for i in range(0,len(calpha_locations)):
                for j in range(i+1,len(calpha_locations)):
                    if(all_residues[i].get_calpha().calculate_distance(all_residues[j].get_calpha()) <= thr ):
                        adj[i][j] = 1
                        adj[j][i] = 1
        
    #Adj to edge_list
    edge_list = []
    edge_weight_list = []
    for numi,i in enumerate(adj):
        count = 0
        for numj, j in enumerate(i):
            if(j!=0.0):
                edge_list.append([numi,numj])
                count+=1

    #Adjacency list format (Might have to change later)
    edge_list = numpy.array(edge_list).T
    edge_weight_list = numpy.array(edge_weight_list)

    return ( splitted_line[0], {'NodeFeatures':node_features, 'edge_list':edge_list, 'Y':y} )