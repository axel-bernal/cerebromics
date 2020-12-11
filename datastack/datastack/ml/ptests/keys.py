
"""
Names of columns we use in regression tests.

The tests here currently are all local, taking the data from files in
the resources directory. They don't rely on a connnection
to vdb or rdb. 

The data in the files originated in Rosetta queries, for the column values
listed here, but with gender mapped to a 0/1 set. As long as we use the
current data files we don't need to worry about chnages in new Rosetta
versions.
"""

key = 'ds.index.sample_key'
gender = 'dynamic.FACE_P.gender.v1.value'
eyes = ['dynamic.FACE.neyecolor.v1_visit1.a', 'dynamic.FACE.neyecolor.v1_visit1.b', 'dynamic.FACE.neyecolor.v1_visit1.l']
bmi = 'dynamic.FACE.pheno.v1.bmi'
left = 'facepheno.hand.strength.left.m1'
right = 'facepheno.hand.strength.right.m1'
health = 'facepheno.health.status'
height = 'facepheno.height'

wcard1 = 'facepheno.hand.strength.*.m1'
wcard2 = '^facepheno.hand.strength.*.m1$'
