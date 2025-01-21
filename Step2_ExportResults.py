try:
    from odbAccess import *
    from abaqusConstants import *
    from symbolicConstants import *
except ImportError:
    print('You need to run these commands via Abaqus CAE kernel')


from collections import namedtuple
import numpy as np
from typing import List,Tuple
from scipy.io import savemat,loadmat
from copy import deepcopy
from helper_functions import *
import pickle
'''
TO DO:
Fix Element Outputs need to average if we stick to only nodes
'''



def get_data(frame,nodeset,field_name,field_comps,field_invar):
    # if frame.fieldOutputs[field_name].locations[0].position == NODAL:
    position = NODAL
    field_output =frame.fieldOutputs[field_name].getSubset(region =nodeset,position = position)
    fieldValues = field_output.values
    frame_data = [ [v.data[field_component] for field_component in field_comps.values()] + [getattr(v,field_invariant) for field_invariant in field_invar]  for v in fieldValues]
    return frame_data



class ODB_Exporter():
    def __init__(self,odb) -> None:
        self.odb = odb

    def add_FieldOutputs(self,field_outputs:dict,step,nodeset):
        '''
        Add Field Outputs requests. This code needs to:
            - check if the requested field is available
            - Grab component labels which are stored in a tuple so indexing is needed
            - Grab field invariants which are strings and the getattr method is needed
            - Concatenate all the output requests into a single array
        
        Returns:
            - data : np.array a single array containing all the 
        '''

        #Check Requested field Output are valid Requests
        data = []
        output_vars = []
        
        frame = step.frames[0]
        for field_name,field_vars in field_outputs.items():

            if field_name not in frame.fieldOutputs.keys():
                raise ValueError(f'{field_name} not a valid field output. Valid Outputs are {list(frame.fieldOutputs.keys())}')
            
            #Check Requested field Output are valid Requests
            field_output =frame.fieldOutputs[field_name]
            field_components = field_output.componentLabels
            # Get the valid field invariants for the field output as a string. Tricky as invariants stored as SymbolicConstant obj not strings.
            field_invariants = [invariant for invariant in field_output.values[0].__members__ if (SymbolicConstant(str.upper(invariant)) in field_output.validInvariants )]
            
            #Check and assign requested field variable if  either component e.g. U1 or invariant E.g. Mises
            field_invar = []
            field_comps = {}
            for i,field_var in enumerate(field_vars):
                
                #Components are accessed by indexing in get_data method
                if field_var in field_components:
                    field_comps[field_var] = i
                
                #Invariants are accessed by getattr methods in get_data method
                elif field_var in field_invariants:
                    field_invar.append(field_var)
                else:
                    raise ValueError(f'{field_var} not a valid request! Request are Case Sensitive! Valid Requests Are {list(field_components) + list(field_invariants)} for Field Output {field_name}')
            print(field_name,field_comps,field_invar)
            field_values = np.array([get_data(frame,nodeset,field_name,field_comps,field_invar) for frame in step.frames],dtype=np.float32)
            
            data.append(field_values)
            output_vars += list(field_comps.keys()) + (field_invar)

        data = np.concatenate(data,axis = -1)
        print(data[0,0])
        return data,output_vars

    def export_results(self,instanceName,field_outputs:dict,stepName,parameters:dict | None = None,export = 'compact'):
        '''
        Get Requested Field output from an ODB file and convert it into a single numpy array across all frames that can easily be converted to a tensor.
        
        This must be run using Abaqus Python via CAE

        - odb: odb Object 
        - partName:str name of part Instance to extract results from
        - field_outputs: dictionary where key = Primary Variable (such as S,U or E) and the value is a list of string of different componants (such as U1,U2 etc) or invariant (such as mises,principal stresses etc)
        - stepName: str the step to request the value over
        - export_type: str the format to output the data
            - 'compact' -> Seperate arrays for coordinates, time and output values. Output array is of the shape (number Frames in step,number of nodes in Part,total number of output requests).
                This is suitable if either performing Graph based deep learning or to keep file size down

            - 'tabular' -> have a single array holding all the data in a single array of shape (num Frames* number of nodes, coords size + 1 + number of output requests) in a tabular/csv style
                This is suitable for parametric, point based deeplearning e.g. where the inputs are x,y,z,t,params. However this need additional space as coordinates and time values are repeated

        Output:
            dict[str,[array,List[str]]]: A dict of dictionary where the first key provide information of the type of tuple being stored. the tuple stored is of size two containing an array 
            and then a list of strings act as a header detailing each column.
            
            'coords': store an array of the coordinates of each node followed by the headers [x,y,z]. None if exporttype == tabular
            'time' : store an array of time points/framevalues in the ODB file. None if exporttype == tabular
            'data': numpy array containing all the data 

        '''
        odb = self.odb
        #Only Export
        partName = instanceName 
        if partName is not None:
            part = odb.rootAssembly.instances[partName]
            elements = part.elements
            nodes = part.nodes
            #First get a list of coordinates of the nodes
            coords = np.array([n.coordinates for n in nodes])
            part_set = f'{partName}_ALL_NODES'
            if part_set not in odb.rootAssembly.nodeSets.keys():
                nodeset = odb.rootAssembly.NodeSet(name = part_set,nodes = (nodes,))
            else:
                nodeset = odb.rootAssembly.nodeSets[part_set]

        step = odb.steps[stepName]
        print(coords[0])
        data,output_vars = self.add_FieldOutputs(field_outputs,step,nodeset)

        cart_csys = ['x','y','z']
        dim = coords.shape[-1]

        time_array = np.array([float(frame.frameValue) for frame in step.frames])
        
        if parameters is None:
            parameters = {}

        #Set up Dictionary and information
        results = {}
        results['part name'] = partName
        results['node labels'] = np.array([n.label for n in nodes])

        results['parameter names'] = list(parameters.keys())
        results['parameters'] = list(parameters.values())
        results['coords'] = coords
        results['coordinate system'] = cart_csys[:dim]
        results['dimension'] = dim
        results['time'] = time_array
        results['data'] = data
        results['output vars'] = output_vars
        results['headers'] = output_vars
        results['input vars'] = []
        results['export'] = 'compact'
        print(results['parameters'])
        if export == 'tabular':
            return convert_compact_to_tabular(results)
        elif export == 'compact':
            return results
        else:
            raise ValueError('variable export only takes strings value tabular or compact')


# def export_results(odb,):

if __name__ == '__main__':
    import glob
    from pathlib import Path
    os.makedirs('ML_Folder',exist_ok=True)
    ps = glob.glob('results/*/crush*.odb')
    print(len(ps))
    for p in ps:
        file = Path(p)
        print(file.name)
        # savemat('tubecrush2.mat',tab)
        params = file.stem.split('_')[1:]

        param_dict = {}
        for i in range(0,6,2):
            key = params[i]
            val = float(params[i+1].replace('-','.'))
            param_dict[key] = val

        odb = openOdb(str(file))
        odbBuilder = ODB_Exporter(odb)
        #,'S':('mises','S11')
        tab = odbBuilder.export_results('TUBE-1',{'U':('U1','U2','U3')},'TubeCrush',parameters = param_dict,export = 'tabular')
        # comp = convert_tabular_to_compact(tab)
        # print(tab['data'].shape)
        savemat(f'ML_Folder/{file.stem}.mat',tab)
        # odb.close()
        # break