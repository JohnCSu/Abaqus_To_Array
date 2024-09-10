from odbAccess import *
from abaqusConstants import *
from symbolicConstants import *
from collections import namedtuple
import numpy as np

# class abaqus_to_numpy():
#     def __init__(self,odbFile) -> None:


def get_data(frame,nodeset,field_outputs,field_comps,field_invar):
    for field_name,field_vars in field_outputs.items():
        field_output =frame.fieldOutputs[field_name].getSubset(region =nodeset,position =NODAL)
        fieldValues = field_output.values
        frame_data = [ [v.data[field_component] for field_component in field_comps.values()] + [getattr(v,field_invariant) for field_invariant in field_invar.keys()]  for v in fieldValues]
        return frame_data



def export_results(odb,partName,field_outputs:dict,stepName,export_type = 'compact'):
    '''
    Get Requested Field output from an ODB file and convert it into a single numpy array across all frames that can easily be converted to a tensor.
    
    This must be run using Abaqus Python via CAE

    - odb: odb Object 
    - partName:str name of part to extract results from
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
        'outputs': if exporttype == compact 

    '''

    #Only Export 
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


    #Check Requested field Output are valid Requests 
    frame = step.frames[0]
    for field_name,field_vars in field_outputs.items():
        field_output =frame.fieldOutputs[field_name]
        field_components = field_output.componentLabels
        # field_invariants =  [str(invariant) for invariant in field_output.validInvariant]
        field_invariants = [invariant for invariant in field_output.values[0].__members__ if (SymbolicConstant(str.upper(invariant)) in field_output.validInvariants )]

        field_invar = {}
        field_comps = {}
        for i,field_var in enumerate(field_vars):
            if field_var in field_components:
                field_comps[field_var] = i
            #Because for some dumb reason we store invarints as symbolic constants rather than just strings...
            elif field_var in field_invariants:
                field_invar[field_var] = i
            else:
                raise ValueError(f'{field_var} not a valid request! Request are Case Sensitive! Valid Requests Are {list(field_components) + list(field_invariants)} for Field Output {field_name}')



    data = np.array([get_data(frame,nodeset,field_outputs,field_comps,field_invar) for frame in step.frames])
    time_array = np.array([float(frame.frameValue) for frame in step.frames])
    headers = ['frame no','node label'] + list(field_comps.keys()) + list(field_invar.keys())

    cart_csys = ['x','y','z']
    dim = coords.shape[-1]

    results = {}
    if export_type == 'compact':
        results['coords'] = (coords,cart_csys[:dim])
        results['time'] = (time_array,['t'])
        results['output'] = (data,headers)

        return results

    elif export_type == 'tabular':
        element_labels = np.array(list(range(coords.shape[0])))[:,np.newaxis]
        frame_csv =  np.concatenate ( [ np.concatenate([element_labels,coords,np.ones((coords.shape[0],1))*t,data[i]],axis = -1) for i,t in enumerate(time_array) ], axis = 0 )
        headers = ['node label',*cart_csys[:dim],'t'] + list(field_comps.keys()) + list(field_invar.keys())
        results['coords'] = None
        results['time'] = None
        results['output'] = (frame_csv,headers)
        
        
        return results

    else:
        raise ValueError('Export Options are compact or tabular ')





if __name__ == '__main__':
    odb = openOdb('TubeCrush.odb')
    x = export_results(odb,'TUBE-1',{'U':('U1','U2','U3')},'TubeCrush',export_type = 'tabular')
    