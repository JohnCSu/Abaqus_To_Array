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
from pathlib import Path
import os
from helper_functions import *


class MLodbBuilder():
    def __init__(self,original_odb:Odb,instanceName:str,nn_odb_name:str|None,nn_data:dict):
        '''

        '''
        self.validInvariants = {
            SCALAR: tuple() ,
            VECTOR: (MAGNITUDE,),
            TENSOR_3D_FULL: (MISES, TRESCA, PRESS, INV3, MAX_PRINCIPAL, MID_PRINCIPAL, MIN_PRINCIPAL),
            TENSOR_3D_SURFACE: (MAX_PRINCIPAL, MIN_PRINCIPAL, MAX_INPLANE_PRINCIPAL, MIN_INPLANE_PRINCIPAL),
            TENSOR_3D_PLANAR:  (MISES, TRESCA, PRESS, INV3, MAX_PRINCIPAL, MID_PRINCIPAL, MIN_PRINCIPAL, MAX_INPLANE_PRINCIPAL, MIN_INPLANE_PRINCIPAL, OUTOFPLANE_PRINCIPAL),
            TENSOR_2D_SURFACE: (MAX_PRINCIPAL, MIN_PRINCIPAL, MAX_INPLANE_PRINCIPAL, MIN_INPLANE_PRINCIPAL),
            TENSOR_2D_PLANAR: (MISES, TRESCA, PRESS, INV3, MAX_PRINCIPAL, MID_PRINCIPAL, MIN_PRINCIPAL, MAX_INPLANE_PRINCIPAL, MIN_INPLANE_PRINCIPAL, OUTOFPLANE_PRINCIPAL),
        }
        self.nn_data = nn_data
        self.node_labels = nn_data['node labels']
        self.time_array:np.ndarray = nn_data['time']


        self.original_odb = original_odb
        # print(save_dir)

        if nn_odb_name is None:
            odb_path = Path(original_odb.name)
            nn_odb_name = f'ML_{odb_path.stem}'
        
        if '.odb' not in nn_odb_name:
            nn_odb_name = nn_odb_name + '.odb'
    
        self.nn_odb = Odb(name = str(nn_odb_name),analysisTitle = f'Neural Network Solution')

        self.nn_odb_save_dir = os.getcwd()


        instance = original_odb.rootAssembly.instances[instanceName]
        in_nodes = instance.nodes
        in_elem = instance.elements
        # Create ODB Part from instance
        part = self.nn_odb.Part(instanceName,embeddedSpace=instance.embeddedSpace, type=instance.type)

        #Add Nodes
        node_labels,node_coords = tuple(zip(*[(n.label,n.coordinates) for n in in_nodes]))
        part.addNodes(labels = node_labels,coordinates = node_coords,nodeSetName='All_Nodes')

        #Add Element Data
        elem_label,elem_con = tuple(zip(*[(e.label,e.connectivity) for e in in_elem]))
        elem_type = in_elem[0].type
        part.addElements(labels = elem_label,connectivity = elem_con,type = elem_type,elementSetName = 'ALL_Elements')

        self.nn_instance = self.nn_odb.rootAssembly.Instance(name= instanceName,object= part)

        
        if 'step-1' not in self.nn_odb.steps.keys():
            self.step = self.nn_odb.Step(name = 'step-1',description='Machine Learning Solution', domain=TIME, timePeriod=self.time_array.max())

        for i,t in enumerate(self.time_array,start = 0):
            self.step.Frame(incrementNumber= i,frameValue=t,description=f'Time {t:.4E}')
        self.update()
        self.save()

    def addDisplacementVector(self,keys,U_name = 'U'):
        '''
        Set the displacement for the ML Solution. treated slightly differently to a regular field as this is used as the deformed variable
        '''
        for key in keys:
            assert key in self.nn_data['output vars']
        assert len(keys) == 3

        uField = self.addField(U_name,keys,VECTOR,description= 'Machine Learning Displacements')
        self.step.setDefaultDeformedField(uField)
        self.update()
        self.nn_odb.save()


    def addField(self,fieldName:str,keys:list[str],dataType:SymbolicConstant,valid_invariants = None,position:SymbolicConstant = NODAL,description = ''):
        if valid_invariants is None:
            valid_invariants = self.validInvariants[dataType]

        indices = [self.nn_data['output vars'].index(key) for key in keys]

        assert len(indices) == len(keys), 'A key was not found in output vars. Please check self.nn_data["output vars"]'

        for i,frame in enumerate(self.step.frames):
            field_data = np.ascontiguousarray(self.nn_data['data'][i,:,indices].transpose())
            # print(field_data.shape,self.node_labels.shape)
            Field = frame.FieldOutput(name = fieldName,description=description, type=dataType,validInvariants = valid_invariants)
            Field.addData(position =position, instance = self.nn_instance,labels = self.node_labels ,data = field_data)
        self.update()
        self.nn_odb.save()
        return Field


    def save(self):
        self.nn_odb.save()

    def close(self):
        self.nn_odb.close()
        self.original_odb.close()
    def update(self):
        self.nn_odb.update()

def get_column_from_array(data_dict,column):
    columns:list = data_dict['headers']
    data = data_dict['data']
    return data[:,columns.index(column)]
if __name__ == '__main__':
    
    mats = glob.glob('Test/crush*.mat')
    # overwrite = True
    for mat in mats:
        file = Path(mat)
        new_odb_name = f'ML_{file.stem}.odb'
        # new_odb_name = 
        nn_data = loaddata(file)
        odb = openOdb(f'results/{file.stem}/{file.stem}.odb')
        if os.path.exists(new_odb_name):
            os.remove(new_odb_name)

        OdbBuilder = MLodbBuilder(odb,'TUBE-1',new_odb_name,nn_data)
        OdbBuilder.addDisplacementVector(['U1','U2','U3'])
        OdbBuilder.close()
        print(f'{new_odb_name} Done!')
    # createODBFromNN(odb,'TUBE-1',nn_data)

