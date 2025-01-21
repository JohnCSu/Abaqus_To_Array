'''
We Examine 2 parameters:

W - Width of Square Tube def 0.1 range 0.075 - 0.2
T - Thickenss of Square Tube  def 0.001 range 0.0005 - 0.0025 


For now V is ignored
V - Velocity of rigid plate def 8.94  range 6 - 11
'''

from abaqus import *
from abaqusConstants import *
from caeModules import *
from driverUtils import executeOnCaeStartup
import math

from helper_functions import *
import time

import os


continue_runs = True
run_start = 0


# else:
parameters = np.loadtxt('parameters.txt',dtype=float,delimiter='\t')
run_start = 0

Num_el = 2000 # Approximate

crush_model_name = 'crush'

os.makedirs('results',exist_ok=True)

#Update Geometry based on parameter Width
for i,(Thickness,Vel,Width) in enumerate(parameters[run_start:],start = run_start):
    print(f'Run {i}:  Width = {Width},Thickness = {Thickness},Vel = {Vel}')
    param_suffix = f"W_{str(Width).replace('.','-')}_T_{str(Thickness).replace('.','-')}_V_{str(Vel).replace('.','-')}"
    Mesh_Seed_Size = math.sqrt(1.6*Width/Num_el) 
    openMdb('Crush.cae')

    p = mdb.models['Buckle'].parts['Tube']
    s = p.features['Shell extrude-1'].sketch
    mdb.models['Buckle'].ConstrainedSketch(name='__edit__', objectToCopy=s)
    s1 = mdb.models['Buckle'].sketches['__edit__']
    g, v, d, c = s1.geometry, s1.vertices, s1.dimensions, s1.constraints
    s1.setPrimaryObject(option=SUPERIMPOSE)
    p.projectReferencesOntoSketch(sketch=s1, 
        upToFeature=p.features['Shell extrude-1'], filter=COPLANAR_EDGES)
    s1.Spot(point=(0.0, 0.0))
    s1.FixedConstraint(entity=v[4])
    s1.EqualDistanceConstraint(entity1=g[2], entity2=g[4], midpoint=v[4])
    s1.EqualDistanceConstraint(entity1=g[3], entity2=g[5], midpoint=v[4])
    w = str(Width)
    s=mdb.models['Buckle'].sketches['__edit__']
    s.parameters['W'].setValues(expression=w)
    s1.unsetPrimaryObject()
    p.features['Shell extrude-1'].setValues(sketch=s1)
    del mdb.models['Buckle'].sketches['__edit__']
    p.regenerate()

    a = mdb.models['Buckle'].rootAssembly
    a.regenerate()

    #Shell Thickness Update
    mdb.models['Buckle'].sections['TubeSection'].setValues(preIntegrate=OFF, 
        material='Steel', thicknessType=UNIFORM, thickness=Thickness, 
        thicknessField='', nodalThicknessField='', idealization=NO_IDEALIZATION, 
        integrationRule=SIMPSON, numIntPts=3)


    #Meshing Update
    p = mdb.models['Buckle'].parts['Tube']
    p.deleteMesh()
    p.seedPart(size=Mesh_Seed_Size, deviationFactor=0.1, minSizeFactor=0.1)
    p = mdb.models['Buckle'].parts['Tube']
    p.generateMesh()
    a = mdb.models['Buckle'].rootAssembly
    a.regenerate()

    mdb.models['Buckle'].steps['TubeBuckle'].setValues(maxIterations=100)

    buckle_job = f"Buckle_{param_suffix}"



    mdb.Job(name=buckle_job, model='Buckle', description='', numCpus=1)
    mdb.jobs[buckle_job].submit(consistencyChecking=OFF)
    mdb.jobs[buckle_job].waitForCompletion()

    #Create Crush Model. Use buckling job as imperfections
    

    if crush_model_name in mdb.models.keys():
        del mdb.models[crush_model_name]
    mdb.Model(name=crush_model_name, objectToCopy=mdb.models['Buckle'])

    #Remove the load from buckle
    del mdb.models[crush_model_name].loads['Load-1']
    a = mdb.models[crush_model_name].rootAssembly
    a.regenerate()
    #Surfaces

    s1 = a.instances['Tube-1'].faces
    side12Faces1 = s1.getSequenceFromMask(mask=('[#f ]', ), )
    a.Surface(side12Faces=side12Faces1, name='Tube')
    #: The surface 'Tube' has been created (4 faces).
    s1 = a.instances['Plate-1'].faces
    side2Faces1 = s1.getSequenceFromMask(mask=('[#1 ]', ), )
    a.Surface(side2Faces=side2Faces1, name='TopSurf')
    #: The surface 'TopSurf' has been created (1 face).


    #Imperfections
    n1 = a.instances['Tube-1'].nodes
    nodes1 = n1.getSequenceFromMask(mask=('[#ffffffff:66 ]', ), )
    a.Set(nodes=nodes1, name='Imperfection_Node_Set')
    region=a.sets['Imperfection_Node_Set']
    mdb.models['crush'].rootAssembly.engineeringFeatures.FileImperfection(
        name='Imperfection-1', file=f'{buckle_job}.odb', step=1, increment=1, 
        linearSuperpositions=((1, 2e-05), (2, 2.5e-06), (3, 2.5e-06), (4, 1.8e-06), 
        (5, 1.6e-06), (6, 1e-06), (7, 1e-06), (8, 8e-07), (9, 2e-07), (10, 2e-07)))


    #Replace Step
    mdb.models['crush'].ExplicitDynamicsStep(name='TubeBuckle', previous='Initial', 
        maintainAttributes=True, timePeriod=0.015, improvedDtMethod=ON)
    mdb.models[crush_model_name].steps.changeKey(fromName='TubeBuckle', toName='TubeCrush')

    mdb.models['crush'].fieldOutputRequests['F-Output-1'].setValues(
    numIntervals=30)

    #Contact
    del mdb.models['crush'].interactions['Int-2']


    mdb.models['crush'].ContactProperty('IntProp-2')
    mdb.models['crush'].interactionProperties['IntProp-2'].TangentialBehavior(
        formulation=PENALTY, directionality=ISOTROPIC, slipRateDependency=OFF, 
        pressureDependency=OFF, temperatureDependency=OFF, dependencies=0, table=((
        0.1, ), ), shearStressLimit=None, maximumElasticSlip=FRACTION, 
        fraction=0.005, elasticSlipStiffness=None)
    mdb.models['crush'].interactionProperties['IntProp-2'].NormalBehavior(
        pressureOverclosure=HARD, allowSeparation=ON, 
        constraintEnforcementMethod=DEFAULT)

    #Set Global Contact
    mdb.models['crush'].ContactExp(name='Int-1', createStepName='Initial')
    mdb.models['crush'].interactions['Int-1'].includedPairs.setValuesInStep(
        stepName='Initial', useAllstar=ON)
    mdb.models['crush'].interactions['Int-1'].contactPropertyAssignments.appendInStep(
        stepName='Initial', assignments=((GLOBAL, SELF, 'IntProp-1'), ))

    # Set Gloval to int prop fith friction 0.1
    mdb.models['crush'].interactions['Int-1'].contactPropertyAssignments.changeValuesInStep(
        stepName='Initial', index=0, value='IntProp-2')
    r11=mdb.models['crush'].rootAssembly.surfaces['TopSurf']
    r12=mdb.models['crush'].rootAssembly.surfaces['Tube']

    #Individiual Contact Tube and Plate
    mdb.models['crush'].interactions['Int-1'].contactPropertyAssignments.appendInStep(
        stepName='Initial', assignments=((r11, r12, 'IntProp-1'), ))
    #Velocity
    region = a.sets['RigidRefBot']
    mdb.models[crush_model_name].Velocity(name='Velocity', region=region, field='', 
        distributionType=MAGNITUDE, velocity1=Vel, omega=0.0)

    a.regenerate()  
    #Submit Crush Job
    crush_job = f"{crush_model_name}_{param_suffix}"
    
    mdb.Job(name=crush_job, model=crush_model_name,numDomains=4,numCpus=4)
    mdb.jobs[crush_job].submit(consistencyChecking=OFF)
    mdb.jobs[crush_job].waitForCompletion()

    time.sleep(5)
    moveFiles(f'results/{crush_job}',f'*{param_suffix}*')




