дя
║Ј
B
AssignVariableOp
resource
value"dtype"
dtypetypeѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Џ
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(ѕ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
Й
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕ
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.4.12v2.4.0-49-g85c8b2a817f8ъа	
Ю
 simple_conv_encoder/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	г*1
shared_name" simple_conv_encoder/dense/kernel
ќ
4simple_conv_encoder/dense/kernel/Read/ReadVariableOpReadVariableOp simple_conv_encoder/dense/kernel*
_output_shapes
:	г*
dtype0
ћ
simple_conv_encoder/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name simple_conv_encoder/dense/bias
Ї
2simple_conv_encoder/dense/bias/Read/ReadVariableOpReadVariableOpsimple_conv_encoder/dense/bias*
_output_shapes
:*
dtype0
б
!simple_conv_encoder/conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!simple_conv_encoder/conv1d/kernel
Џ
5simple_conv_encoder/conv1d/kernel/Read/ReadVariableOpReadVariableOp!simple_conv_encoder/conv1d/kernel*"
_output_shapes
:@*
dtype0
ќ
simple_conv_encoder/conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*0
shared_name!simple_conv_encoder/conv1d/bias
Ј
3simple_conv_encoder/conv1d/bias/Read/ReadVariableOpReadVariableOpsimple_conv_encoder/conv1d/bias*
_output_shapes
:@*
dtype0
д
#simple_conv_encoder/conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#simple_conv_encoder/conv1d_1/kernel
Ъ
7simple_conv_encoder/conv1d_1/kernel/Read/ReadVariableOpReadVariableOp#simple_conv_encoder/conv1d_1/kernel*"
_output_shapes
:@@*
dtype0
џ
!simple_conv_encoder/conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!simple_conv_encoder/conv1d_1/bias
Њ
5simple_conv_encoder/conv1d_1/bias/Read/ReadVariableOpReadVariableOp!simple_conv_encoder/conv1d_1/bias*
_output_shapes
:@*
dtype0
д
#simple_conv_encoder/conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#simple_conv_encoder/conv1d_2/kernel
Ъ
7simple_conv_encoder/conv1d_2/kernel/Read/ReadVariableOpReadVariableOp#simple_conv_encoder/conv1d_2/kernel*"
_output_shapes
:@@*
dtype0
џ
!simple_conv_encoder/conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!simple_conv_encoder/conv1d_2/bias
Њ
5simple_conv_encoder/conv1d_2/bias/Read/ReadVariableOpReadVariableOp!simple_conv_encoder/conv1d_2/bias*
_output_shapes
:@*
dtype0
д
#simple_conv_encoder/conv1d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#simple_conv_encoder/conv1d_3/kernel
Ъ
7simple_conv_encoder/conv1d_3/kernel/Read/ReadVariableOpReadVariableOp#simple_conv_encoder/conv1d_3/kernel*"
_output_shapes
:@@*
dtype0
џ
!simple_conv_encoder/conv1d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!simple_conv_encoder/conv1d_3/bias
Њ
5simple_conv_encoder/conv1d_3/bias/Read/ReadVariableOpReadVariableOp!simple_conv_encoder/conv1d_3/bias*
_output_shapes
:@*
dtype0
д
#simple_conv_encoder/conv1d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#simple_conv_encoder/conv1d_4/kernel
Ъ
7simple_conv_encoder/conv1d_4/kernel/Read/ReadVariableOpReadVariableOp#simple_conv_encoder/conv1d_4/kernel*"
_output_shapes
:@@*
dtype0
џ
!simple_conv_encoder/conv1d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!simple_conv_encoder/conv1d_4/bias
Њ
5simple_conv_encoder/conv1d_4/bias/Read/ReadVariableOpReadVariableOp!simple_conv_encoder/conv1d_4/bias*
_output_shapes
:@*
dtype0
д
#simple_conv_encoder/conv1d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*4
shared_name%#simple_conv_encoder/conv1d_5/kernel
Ъ
7simple_conv_encoder/conv1d_5/kernel/Read/ReadVariableOpReadVariableOp#simple_conv_encoder/conv1d_5/kernel*"
_output_shapes
:@@*
dtype0
џ
!simple_conv_encoder/conv1d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!simple_conv_encoder/conv1d_5/bias
Њ
5simple_conv_encoder/conv1d_5/bias/Read/ReadVariableOpReadVariableOp!simple_conv_encoder/conv1d_5/bias*
_output_shapes
:@*
dtype0
д
#simple_conv_encoder/conv1d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *4
shared_name%#simple_conv_encoder/conv1d_6/kernel
Ъ
7simple_conv_encoder/conv1d_6/kernel/Read/ReadVariableOpReadVariableOp#simple_conv_encoder/conv1d_6/kernel*"
_output_shapes
:@ *
dtype0
џ
!simple_conv_encoder/conv1d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!simple_conv_encoder/conv1d_6/bias
Њ
5simple_conv_encoder/conv1d_6/bias/Read/ReadVariableOpReadVariableOp!simple_conv_encoder/conv1d_6/bias*
_output_shapes
: *
dtype0
д
#simple_conv_encoder/conv1d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#simple_conv_encoder/conv1d_7/kernel
Ъ
7simple_conv_encoder/conv1d_7/kernel/Read/ReadVariableOpReadVariableOp#simple_conv_encoder/conv1d_7/kernel*"
_output_shapes
: *
dtype0
џ
!simple_conv_encoder/conv1d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!simple_conv_encoder/conv1d_7/bias
Њ
5simple_conv_encoder/conv1d_7/bias/Read/ReadVariableOpReadVariableOp!simple_conv_encoder/conv1d_7/bias*
_output_shapes
:*
dtype0
д
#simple_conv_encoder/conv1d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#simple_conv_encoder/conv1d_8/kernel
Ъ
7simple_conv_encoder/conv1d_8/kernel/Read/ReadVariableOpReadVariableOp#simple_conv_encoder/conv1d_8/kernel*"
_output_shapes
:*
dtype0
џ
!simple_conv_encoder/conv1d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!simple_conv_encoder/conv1d_8/bias
Њ
5simple_conv_encoder/conv1d_8/bias/Read/ReadVariableOpReadVariableOp!simple_conv_encoder/conv1d_8/bias*
_output_shapes
:*
dtype0
д
#simple_conv_encoder/conv1d_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#simple_conv_encoder/conv1d_9/kernel
Ъ
7simple_conv_encoder/conv1d_9/kernel/Read/ReadVariableOpReadVariableOp#simple_conv_encoder/conv1d_9/kernel*"
_output_shapes
:*
dtype0
џ
!simple_conv_encoder/conv1d_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!simple_conv_encoder/conv1d_9/bias
Њ
5simple_conv_encoder/conv1d_9/bias/Read/ReadVariableOpReadVariableOp!simple_conv_encoder/conv1d_9/bias*
_output_shapes
:*
dtype0
е
$simple_conv_encoder/conv1d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$simple_conv_encoder/conv1d_10/kernel
А
8simple_conv_encoder/conv1d_10/kernel/Read/ReadVariableOpReadVariableOp$simple_conv_encoder/conv1d_10/kernel*"
_output_shapes
:*
dtype0
ю
"simple_conv_encoder/conv1d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"simple_conv_encoder/conv1d_10/bias
Ћ
6simple_conv_encoder/conv1d_10/bias/Read/ReadVariableOpReadVariableOp"simple_conv_encoder/conv1d_10/bias*
_output_shapes
:*
dtype0

NoOpNoOp
Џ<
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*о;
value╠;B╔; B┬;
Ѓ
	convs
flatten
out
regularization_losses
trainable_variables
	variables
	keras_api

signatures
N
	0

1
2
3
4
5
6
7
8
9
10
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
 
Х
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10
)11
*12
+13
,14
-15
.16
/17
018
119
220
321
22
23
Х
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10
)11
*12
+13
,14
-15
.16
/17
018
119
220
321
22
23
Г
4metrics
regularization_losses
5layer_regularization_losses
trainable_variables
6layer_metrics

7layers
8non_trainable_variables
	variables
 
h

kernel
bias
9regularization_losses
:trainable_variables
;	variables
<	keras_api
h

 kernel
!bias
=regularization_losses
>trainable_variables
?	variables
@	keras_api
h

"kernel
#bias
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
h

$kernel
%bias
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
h

&kernel
'bias
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
h

(kernel
)bias
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
h

*kernel
+bias
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
h

,kernel
-bias
Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
h

.kernel
/bias
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
h

0kernel
1bias
]regularization_losses
^trainable_variables
_	variables
`	keras_api
h

2kernel
3bias
aregularization_losses
btrainable_variables
c	variables
d	keras_api
 
 
 
Г
emetrics
regularization_losses
flayer_regularization_losses
trainable_variables
glayer_metrics

hlayers
inon_trainable_variables
	variables
[Y
VARIABLE_VALUE simple_conv_encoder/dense/kernel%out/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEsimple_conv_encoder/dense/bias#out/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
Г
jmetrics
regularization_losses
klayer_regularization_losses
trainable_variables
llayer_metrics

mlayers
nnon_trainable_variables
	variables
ge
VARIABLE_VALUE!simple_conv_encoder/conv1d/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEsimple_conv_encoder/conv1d/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#simple_conv_encoder/conv1d_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE!simple_conv_encoder/conv1d_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#simple_conv_encoder/conv1d_2/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE!simple_conv_encoder/conv1d_2/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#simple_conv_encoder/conv1d_3/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE!simple_conv_encoder/conv1d_3/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#simple_conv_encoder/conv1d_4/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE!simple_conv_encoder/conv1d_4/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#simple_conv_encoder/conv1d_5/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!simple_conv_encoder/conv1d_5/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#simple_conv_encoder/conv1d_6/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!simple_conv_encoder/conv1d_6/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#simple_conv_encoder/conv1d_7/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!simple_conv_encoder/conv1d_7/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#simple_conv_encoder/conv1d_8/kernel1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!simple_conv_encoder/conv1d_8/bias1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#simple_conv_encoder/conv1d_9/kernel1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE!simple_conv_encoder/conv1d_9/bias1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE$simple_conv_encoder/conv1d_10/kernel1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE"simple_conv_encoder/conv1d_10/bias1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
^
	0

1
2
3
4
5
6
7
8
9
10
11
12
 
 

0
1

0
1
Г
ometrics
9regularization_losses
player_regularization_losses
:trainable_variables
qlayer_metrics

rlayers
snon_trainable_variables
;	variables
 

 0
!1

 0
!1
Г
tmetrics
=regularization_losses
ulayer_regularization_losses
>trainable_variables
vlayer_metrics

wlayers
xnon_trainable_variables
?	variables
 

"0
#1

"0
#1
Г
ymetrics
Aregularization_losses
zlayer_regularization_losses
Btrainable_variables
{layer_metrics

|layers
}non_trainable_variables
C	variables
 

$0
%1

$0
%1
░
~metrics
Eregularization_losses
layer_regularization_losses
Ftrainable_variables
ђlayer_metrics
Ђlayers
ѓnon_trainable_variables
G	variables
 

&0
'1

&0
'1
▓
Ѓmetrics
Iregularization_losses
 ёlayer_regularization_losses
Jtrainable_variables
Ёlayer_metrics
єlayers
Єnon_trainable_variables
K	variables
 

(0
)1

(0
)1
▓
ѕmetrics
Mregularization_losses
 Ѕlayer_regularization_losses
Ntrainable_variables
іlayer_metrics
Іlayers
їnon_trainable_variables
O	variables
 

*0
+1

*0
+1
▓
Їmetrics
Qregularization_losses
 јlayer_regularization_losses
Rtrainable_variables
Јlayer_metrics
љlayers
Љnon_trainable_variables
S	variables
 

,0
-1

,0
-1
▓
њmetrics
Uregularization_losses
 Њlayer_regularization_losses
Vtrainable_variables
ћlayer_metrics
Ћlayers
ќnon_trainable_variables
W	variables
 

.0
/1

.0
/1
▓
Ќmetrics
Yregularization_losses
 ўlayer_regularization_losses
Ztrainable_variables
Ўlayer_metrics
џlayers
Џnon_trainable_variables
[	variables
 

00
11

00
11
▓
юmetrics
]regularization_losses
 Юlayer_regularization_losses
^trainable_variables
ъlayer_metrics
Ъlayers
аnon_trainable_variables
_	variables
 

20
31

20
31
▓
Аmetrics
aregularization_losses
 бlayer_regularization_losses
btrainable_variables
Бlayer_metrics
цlayers
Цnon_trainable_variables
c	variables
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
ё
serving_default_input_1Placeholder*,
_output_shapes
:         ќ*
dtype0*!
shape:         ќ
╣	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1!simple_conv_encoder/conv1d/kernelsimple_conv_encoder/conv1d/bias#simple_conv_encoder/conv1d_1/kernel!simple_conv_encoder/conv1d_1/bias#simple_conv_encoder/conv1d_2/kernel!simple_conv_encoder/conv1d_2/bias#simple_conv_encoder/conv1d_3/kernel!simple_conv_encoder/conv1d_3/bias#simple_conv_encoder/conv1d_4/kernel!simple_conv_encoder/conv1d_4/bias#simple_conv_encoder/conv1d_5/kernel!simple_conv_encoder/conv1d_5/bias#simple_conv_encoder/conv1d_6/kernel!simple_conv_encoder/conv1d_6/bias#simple_conv_encoder/conv1d_7/kernel!simple_conv_encoder/conv1d_7/bias#simple_conv_encoder/conv1d_8/kernel!simple_conv_encoder/conv1d_8/bias#simple_conv_encoder/conv1d_9/kernel!simple_conv_encoder/conv1d_9/bias$simple_conv_encoder/conv1d_10/kernel"simple_conv_encoder/conv1d_10/bias simple_conv_encoder/dense/kernelsimple_conv_encoder/dense/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *,
f'R%
#__inference_signature_wrapper_13153
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ж
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename4simple_conv_encoder/dense/kernel/Read/ReadVariableOp2simple_conv_encoder/dense/bias/Read/ReadVariableOp5simple_conv_encoder/conv1d/kernel/Read/ReadVariableOp3simple_conv_encoder/conv1d/bias/Read/ReadVariableOp7simple_conv_encoder/conv1d_1/kernel/Read/ReadVariableOp5simple_conv_encoder/conv1d_1/bias/Read/ReadVariableOp7simple_conv_encoder/conv1d_2/kernel/Read/ReadVariableOp5simple_conv_encoder/conv1d_2/bias/Read/ReadVariableOp7simple_conv_encoder/conv1d_3/kernel/Read/ReadVariableOp5simple_conv_encoder/conv1d_3/bias/Read/ReadVariableOp7simple_conv_encoder/conv1d_4/kernel/Read/ReadVariableOp5simple_conv_encoder/conv1d_4/bias/Read/ReadVariableOp7simple_conv_encoder/conv1d_5/kernel/Read/ReadVariableOp5simple_conv_encoder/conv1d_5/bias/Read/ReadVariableOp7simple_conv_encoder/conv1d_6/kernel/Read/ReadVariableOp5simple_conv_encoder/conv1d_6/bias/Read/ReadVariableOp7simple_conv_encoder/conv1d_7/kernel/Read/ReadVariableOp5simple_conv_encoder/conv1d_7/bias/Read/ReadVariableOp7simple_conv_encoder/conv1d_8/kernel/Read/ReadVariableOp5simple_conv_encoder/conv1d_8/bias/Read/ReadVariableOp7simple_conv_encoder/conv1d_9/kernel/Read/ReadVariableOp5simple_conv_encoder/conv1d_9/bias/Read/ReadVariableOp8simple_conv_encoder/conv1d_10/kernel/Read/ReadVariableOp6simple_conv_encoder/conv1d_10/bias/Read/ReadVariableOpConst*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *'
f"R 
__inference__traced_save_13554
Ё	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename simple_conv_encoder/dense/kernelsimple_conv_encoder/dense/bias!simple_conv_encoder/conv1d/kernelsimple_conv_encoder/conv1d/bias#simple_conv_encoder/conv1d_1/kernel!simple_conv_encoder/conv1d_1/bias#simple_conv_encoder/conv1d_2/kernel!simple_conv_encoder/conv1d_2/bias#simple_conv_encoder/conv1d_3/kernel!simple_conv_encoder/conv1d_3/bias#simple_conv_encoder/conv1d_4/kernel!simple_conv_encoder/conv1d_4/bias#simple_conv_encoder/conv1d_5/kernel!simple_conv_encoder/conv1d_5/bias#simple_conv_encoder/conv1d_6/kernel!simple_conv_encoder/conv1d_6/bias#simple_conv_encoder/conv1d_7/kernel!simple_conv_encoder/conv1d_7/bias#simple_conv_encoder/conv1d_8/kernel!simple_conv_encoder/conv1d_8/bias#simple_conv_encoder/conv1d_9/kernel!simple_conv_encoder/conv1d_9/bias$simple_conv_encoder/conv1d_10/kernel"simple_conv_encoder/conv1d_10/bias*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference__traced_restore_13636юЄ
Ј
Ш
C__inference_conv1d_1_layer_call_and_return_conditional_losses_13225

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ@*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ќ@*
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ@2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ќ@2
TanhЪ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:         ќ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ќ@
 
_user_specified_nameinputs
љ
э
D__inference_conv1d_10_layer_call_and_return_conditional_losses_13450

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ќ*
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ќ2
TanhЪ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:         ќ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ќ
 
_user_specified_nameinputs
┤
^
B__inference_flatten_layer_call_and_return_conditional_losses_13159

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    ,  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         г2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         г2

Identity"
identityIdentity:output:0*+
_input_shapes
:         ќ:T P
,
_output_shapes
:         ќ
 
_user_specified_nameinputs
с	
┘
@__inference_dense_layer_call_and_return_conditional_losses_13027

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	г*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:         2
TanhЇ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         г::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         г
 
_user_specified_nameinputs
Ј
Ш
C__inference_conv1d_8_layer_call_and_return_conditional_losses_13400

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ќ*
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ќ2
TanhЪ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:         ќ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ќ
 
_user_specified_nameinputs
Ї
З
A__inference_conv1d_layer_call_and_return_conditional_losses_13200

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ@*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ќ@*
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ@2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ќ@2
TanhЪ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:         ќ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ќ
 
_user_specified_nameinputs
Єў
Ш
 __inference__wrapped_model_12646
input_1J
Fsimple_conv_encoder_conv1d_conv1d_expanddims_1_readvariableop_resource>
:simple_conv_encoder_conv1d_biasadd_readvariableop_resourceL
Hsimple_conv_encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource@
<simple_conv_encoder_conv1d_1_biasadd_readvariableop_resourceL
Hsimple_conv_encoder_conv1d_2_conv1d_expanddims_1_readvariableop_resource@
<simple_conv_encoder_conv1d_2_biasadd_readvariableop_resourceL
Hsimple_conv_encoder_conv1d_3_conv1d_expanddims_1_readvariableop_resource@
<simple_conv_encoder_conv1d_3_biasadd_readvariableop_resourceL
Hsimple_conv_encoder_conv1d_4_conv1d_expanddims_1_readvariableop_resource@
<simple_conv_encoder_conv1d_4_biasadd_readvariableop_resourceL
Hsimple_conv_encoder_conv1d_5_conv1d_expanddims_1_readvariableop_resource@
<simple_conv_encoder_conv1d_5_biasadd_readvariableop_resourceL
Hsimple_conv_encoder_conv1d_6_conv1d_expanddims_1_readvariableop_resource@
<simple_conv_encoder_conv1d_6_biasadd_readvariableop_resourceL
Hsimple_conv_encoder_conv1d_7_conv1d_expanddims_1_readvariableop_resource@
<simple_conv_encoder_conv1d_7_biasadd_readvariableop_resourceL
Hsimple_conv_encoder_conv1d_8_conv1d_expanddims_1_readvariableop_resource@
<simple_conv_encoder_conv1d_8_biasadd_readvariableop_resourceL
Hsimple_conv_encoder_conv1d_9_conv1d_expanddims_1_readvariableop_resource@
<simple_conv_encoder_conv1d_9_biasadd_readvariableop_resourceM
Isimple_conv_encoder_conv1d_10_conv1d_expanddims_1_readvariableop_resourceA
=simple_conv_encoder_conv1d_10_biasadd_readvariableop_resource<
8simple_conv_encoder_dense_matmul_readvariableop_resource=
9simple_conv_encoder_dense_biasadd_readvariableop_resource
identityѕб1simple_conv_encoder/conv1d/BiasAdd/ReadVariableOpб=simple_conv_encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOpб3simple_conv_encoder/conv1d_1/BiasAdd/ReadVariableOpб?simple_conv_encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpб4simple_conv_encoder/conv1d_10/BiasAdd/ReadVariableOpб@simple_conv_encoder/conv1d_10/conv1d/ExpandDims_1/ReadVariableOpб3simple_conv_encoder/conv1d_2/BiasAdd/ReadVariableOpб?simple_conv_encoder/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpб3simple_conv_encoder/conv1d_3/BiasAdd/ReadVariableOpб?simple_conv_encoder/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpб3simple_conv_encoder/conv1d_4/BiasAdd/ReadVariableOpб?simple_conv_encoder/conv1d_4/conv1d/ExpandDims_1/ReadVariableOpб3simple_conv_encoder/conv1d_5/BiasAdd/ReadVariableOpб?simple_conv_encoder/conv1d_5/conv1d/ExpandDims_1/ReadVariableOpб3simple_conv_encoder/conv1d_6/BiasAdd/ReadVariableOpб?simple_conv_encoder/conv1d_6/conv1d/ExpandDims_1/ReadVariableOpб3simple_conv_encoder/conv1d_7/BiasAdd/ReadVariableOpб?simple_conv_encoder/conv1d_7/conv1d/ExpandDims_1/ReadVariableOpб3simple_conv_encoder/conv1d_8/BiasAdd/ReadVariableOpб?simple_conv_encoder/conv1d_8/conv1d/ExpandDims_1/ReadVariableOpб3simple_conv_encoder/conv1d_9/BiasAdd/ReadVariableOpб?simple_conv_encoder/conv1d_9/conv1d/ExpandDims_1/ReadVariableOpб0simple_conv_encoder/dense/BiasAdd/ReadVariableOpб/simple_conv_encoder/dense/MatMul/ReadVariableOp»
0simple_conv_encoder/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        22
0simple_conv_encoder/conv1d/conv1d/ExpandDims/dimж
,simple_conv_encoder/conv1d/conv1d/ExpandDims
ExpandDimsinput_19simple_conv_encoder/conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ2.
,simple_conv_encoder/conv1d/conv1d/ExpandDimsЅ
=simple_conv_encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpFsimple_conv_encoder_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02?
=simple_conv_encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOpф
2simple_conv_encoder/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 24
2simple_conv_encoder/conv1d/conv1d/ExpandDims_1/dimБ
.simple_conv_encoder/conv1d/conv1d/ExpandDims_1
ExpandDimsEsimple_conv_encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0;simple_conv_encoder/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@20
.simple_conv_encoder/conv1d/conv1d/ExpandDims_1Б
!simple_conv_encoder/conv1d/conv1dConv2D5simple_conv_encoder/conv1d/conv1d/ExpandDims:output:07simple_conv_encoder/conv1d/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ@*
paddingSAME*
strides
2#
!simple_conv_encoder/conv1d/conv1dС
)simple_conv_encoder/conv1d/conv1d/SqueezeSqueeze*simple_conv_encoder/conv1d/conv1d:output:0*
T0*,
_output_shapes
:         ќ@*
squeeze_dims

§        2+
)simple_conv_encoder/conv1d/conv1d/SqueezeП
1simple_conv_encoder/conv1d/BiasAdd/ReadVariableOpReadVariableOp:simple_conv_encoder_conv1d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype023
1simple_conv_encoder/conv1d/BiasAdd/ReadVariableOpщ
"simple_conv_encoder/conv1d/BiasAddBiasAdd2simple_conv_encoder/conv1d/conv1d/Squeeze:output:09simple_conv_encoder/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ@2$
"simple_conv_encoder/conv1d/BiasAdd«
simple_conv_encoder/conv1d/TanhTanh+simple_conv_encoder/conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:         ќ@2!
simple_conv_encoder/conv1d/Tanh│
2simple_conv_encoder/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        24
2simple_conv_encoder/conv1d_1/conv1d/ExpandDims/dimІ
.simple_conv_encoder/conv1d_1/conv1d/ExpandDims
ExpandDims#simple_conv_encoder/conv1d/Tanh:y:0;simple_conv_encoder/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ@20
.simple_conv_encoder/conv1d_1/conv1d/ExpandDimsЈ
?simple_conv_encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpHsimple_conv_encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02A
?simple_conv_encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp«
4simple_conv_encoder/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4simple_conv_encoder/conv1d_1/conv1d/ExpandDims_1/dimФ
0simple_conv_encoder/conv1d_1/conv1d/ExpandDims_1
ExpandDimsGsimple_conv_encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0=simple_conv_encoder/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@22
0simple_conv_encoder/conv1d_1/conv1d/ExpandDims_1Ф
#simple_conv_encoder/conv1d_1/conv1dConv2D7simple_conv_encoder/conv1d_1/conv1d/ExpandDims:output:09simple_conv_encoder/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ@*
paddingSAME*
strides
2%
#simple_conv_encoder/conv1d_1/conv1dЖ
+simple_conv_encoder/conv1d_1/conv1d/SqueezeSqueeze,simple_conv_encoder/conv1d_1/conv1d:output:0*
T0*,
_output_shapes
:         ќ@*
squeeze_dims

§        2-
+simple_conv_encoder/conv1d_1/conv1d/Squeezeс
3simple_conv_encoder/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp<simple_conv_encoder_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype025
3simple_conv_encoder/conv1d_1/BiasAdd/ReadVariableOpЂ
$simple_conv_encoder/conv1d_1/BiasAddBiasAdd4simple_conv_encoder/conv1d_1/conv1d/Squeeze:output:0;simple_conv_encoder/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ@2&
$simple_conv_encoder/conv1d_1/BiasAdd┤
!simple_conv_encoder/conv1d_1/TanhTanh-simple_conv_encoder/conv1d_1/BiasAdd:output:0*
T0*,
_output_shapes
:         ќ@2#
!simple_conv_encoder/conv1d_1/Tanh│
2simple_conv_encoder/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        24
2simple_conv_encoder/conv1d_2/conv1d/ExpandDims/dimЇ
.simple_conv_encoder/conv1d_2/conv1d/ExpandDims
ExpandDims%simple_conv_encoder/conv1d_1/Tanh:y:0;simple_conv_encoder/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ@20
.simple_conv_encoder/conv1d_2/conv1d/ExpandDimsЈ
?simple_conv_encoder/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpHsimple_conv_encoder_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02A
?simple_conv_encoder/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp«
4simple_conv_encoder/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4simple_conv_encoder/conv1d_2/conv1d/ExpandDims_1/dimФ
0simple_conv_encoder/conv1d_2/conv1d/ExpandDims_1
ExpandDimsGsimple_conv_encoder/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0=simple_conv_encoder/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@22
0simple_conv_encoder/conv1d_2/conv1d/ExpandDims_1Ф
#simple_conv_encoder/conv1d_2/conv1dConv2D7simple_conv_encoder/conv1d_2/conv1d/ExpandDims:output:09simple_conv_encoder/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ@*
paddingSAME*
strides
2%
#simple_conv_encoder/conv1d_2/conv1dЖ
+simple_conv_encoder/conv1d_2/conv1d/SqueezeSqueeze,simple_conv_encoder/conv1d_2/conv1d:output:0*
T0*,
_output_shapes
:         ќ@*
squeeze_dims

§        2-
+simple_conv_encoder/conv1d_2/conv1d/Squeezeс
3simple_conv_encoder/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp<simple_conv_encoder_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype025
3simple_conv_encoder/conv1d_2/BiasAdd/ReadVariableOpЂ
$simple_conv_encoder/conv1d_2/BiasAddBiasAdd4simple_conv_encoder/conv1d_2/conv1d/Squeeze:output:0;simple_conv_encoder/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ@2&
$simple_conv_encoder/conv1d_2/BiasAdd┤
!simple_conv_encoder/conv1d_2/TanhTanh-simple_conv_encoder/conv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:         ќ@2#
!simple_conv_encoder/conv1d_2/Tanh│
2simple_conv_encoder/conv1d_3/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        24
2simple_conv_encoder/conv1d_3/conv1d/ExpandDims/dimЇ
.simple_conv_encoder/conv1d_3/conv1d/ExpandDims
ExpandDims%simple_conv_encoder/conv1d_2/Tanh:y:0;simple_conv_encoder/conv1d_3/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ@20
.simple_conv_encoder/conv1d_3/conv1d/ExpandDimsЈ
?simple_conv_encoder/conv1d_3/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpHsimple_conv_encoder_conv1d_3_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02A
?simple_conv_encoder/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp«
4simple_conv_encoder/conv1d_3/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4simple_conv_encoder/conv1d_3/conv1d/ExpandDims_1/dimФ
0simple_conv_encoder/conv1d_3/conv1d/ExpandDims_1
ExpandDimsGsimple_conv_encoder/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp:value:0=simple_conv_encoder/conv1d_3/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@22
0simple_conv_encoder/conv1d_3/conv1d/ExpandDims_1Ф
#simple_conv_encoder/conv1d_3/conv1dConv2D7simple_conv_encoder/conv1d_3/conv1d/ExpandDims:output:09simple_conv_encoder/conv1d_3/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ@*
paddingSAME*
strides
2%
#simple_conv_encoder/conv1d_3/conv1dЖ
+simple_conv_encoder/conv1d_3/conv1d/SqueezeSqueeze,simple_conv_encoder/conv1d_3/conv1d:output:0*
T0*,
_output_shapes
:         ќ@*
squeeze_dims

§        2-
+simple_conv_encoder/conv1d_3/conv1d/Squeezeс
3simple_conv_encoder/conv1d_3/BiasAdd/ReadVariableOpReadVariableOp<simple_conv_encoder_conv1d_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype025
3simple_conv_encoder/conv1d_3/BiasAdd/ReadVariableOpЂ
$simple_conv_encoder/conv1d_3/BiasAddBiasAdd4simple_conv_encoder/conv1d_3/conv1d/Squeeze:output:0;simple_conv_encoder/conv1d_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ@2&
$simple_conv_encoder/conv1d_3/BiasAdd┤
!simple_conv_encoder/conv1d_3/TanhTanh-simple_conv_encoder/conv1d_3/BiasAdd:output:0*
T0*,
_output_shapes
:         ќ@2#
!simple_conv_encoder/conv1d_3/Tanh│
2simple_conv_encoder/conv1d_4/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        24
2simple_conv_encoder/conv1d_4/conv1d/ExpandDims/dimЇ
.simple_conv_encoder/conv1d_4/conv1d/ExpandDims
ExpandDims%simple_conv_encoder/conv1d_3/Tanh:y:0;simple_conv_encoder/conv1d_4/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ@20
.simple_conv_encoder/conv1d_4/conv1d/ExpandDimsЈ
?simple_conv_encoder/conv1d_4/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpHsimple_conv_encoder_conv1d_4_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02A
?simple_conv_encoder/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp«
4simple_conv_encoder/conv1d_4/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4simple_conv_encoder/conv1d_4/conv1d/ExpandDims_1/dimФ
0simple_conv_encoder/conv1d_4/conv1d/ExpandDims_1
ExpandDimsGsimple_conv_encoder/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp:value:0=simple_conv_encoder/conv1d_4/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@22
0simple_conv_encoder/conv1d_4/conv1d/ExpandDims_1Ф
#simple_conv_encoder/conv1d_4/conv1dConv2D7simple_conv_encoder/conv1d_4/conv1d/ExpandDims:output:09simple_conv_encoder/conv1d_4/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ@*
paddingSAME*
strides
2%
#simple_conv_encoder/conv1d_4/conv1dЖ
+simple_conv_encoder/conv1d_4/conv1d/SqueezeSqueeze,simple_conv_encoder/conv1d_4/conv1d:output:0*
T0*,
_output_shapes
:         ќ@*
squeeze_dims

§        2-
+simple_conv_encoder/conv1d_4/conv1d/Squeezeс
3simple_conv_encoder/conv1d_4/BiasAdd/ReadVariableOpReadVariableOp<simple_conv_encoder_conv1d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype025
3simple_conv_encoder/conv1d_4/BiasAdd/ReadVariableOpЂ
$simple_conv_encoder/conv1d_4/BiasAddBiasAdd4simple_conv_encoder/conv1d_4/conv1d/Squeeze:output:0;simple_conv_encoder/conv1d_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ@2&
$simple_conv_encoder/conv1d_4/BiasAdd┤
!simple_conv_encoder/conv1d_4/TanhTanh-simple_conv_encoder/conv1d_4/BiasAdd:output:0*
T0*,
_output_shapes
:         ќ@2#
!simple_conv_encoder/conv1d_4/Tanh│
2simple_conv_encoder/conv1d_5/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        24
2simple_conv_encoder/conv1d_5/conv1d/ExpandDims/dimЇ
.simple_conv_encoder/conv1d_5/conv1d/ExpandDims
ExpandDims%simple_conv_encoder/conv1d_4/Tanh:y:0;simple_conv_encoder/conv1d_5/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ@20
.simple_conv_encoder/conv1d_5/conv1d/ExpandDimsЈ
?simple_conv_encoder/conv1d_5/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpHsimple_conv_encoder_conv1d_5_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02A
?simple_conv_encoder/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp«
4simple_conv_encoder/conv1d_5/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4simple_conv_encoder/conv1d_5/conv1d/ExpandDims_1/dimФ
0simple_conv_encoder/conv1d_5/conv1d/ExpandDims_1
ExpandDimsGsimple_conv_encoder/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp:value:0=simple_conv_encoder/conv1d_5/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@22
0simple_conv_encoder/conv1d_5/conv1d/ExpandDims_1Ф
#simple_conv_encoder/conv1d_5/conv1dConv2D7simple_conv_encoder/conv1d_5/conv1d/ExpandDims:output:09simple_conv_encoder/conv1d_5/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ@*
paddingSAME*
strides
2%
#simple_conv_encoder/conv1d_5/conv1dЖ
+simple_conv_encoder/conv1d_5/conv1d/SqueezeSqueeze,simple_conv_encoder/conv1d_5/conv1d:output:0*
T0*,
_output_shapes
:         ќ@*
squeeze_dims

§        2-
+simple_conv_encoder/conv1d_5/conv1d/Squeezeс
3simple_conv_encoder/conv1d_5/BiasAdd/ReadVariableOpReadVariableOp<simple_conv_encoder_conv1d_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype025
3simple_conv_encoder/conv1d_5/BiasAdd/ReadVariableOpЂ
$simple_conv_encoder/conv1d_5/BiasAddBiasAdd4simple_conv_encoder/conv1d_5/conv1d/Squeeze:output:0;simple_conv_encoder/conv1d_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ@2&
$simple_conv_encoder/conv1d_5/BiasAdd┤
!simple_conv_encoder/conv1d_5/TanhTanh-simple_conv_encoder/conv1d_5/BiasAdd:output:0*
T0*,
_output_shapes
:         ќ@2#
!simple_conv_encoder/conv1d_5/Tanh│
2simple_conv_encoder/conv1d_6/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        24
2simple_conv_encoder/conv1d_6/conv1d/ExpandDims/dimЇ
.simple_conv_encoder/conv1d_6/conv1d/ExpandDims
ExpandDims%simple_conv_encoder/conv1d_5/Tanh:y:0;simple_conv_encoder/conv1d_6/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ@20
.simple_conv_encoder/conv1d_6/conv1d/ExpandDimsЈ
?simple_conv_encoder/conv1d_6/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpHsimple_conv_encoder_conv1d_6_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02A
?simple_conv_encoder/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp«
4simple_conv_encoder/conv1d_6/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4simple_conv_encoder/conv1d_6/conv1d/ExpandDims_1/dimФ
0simple_conv_encoder/conv1d_6/conv1d/ExpandDims_1
ExpandDimsGsimple_conv_encoder/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp:value:0=simple_conv_encoder/conv1d_6/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 22
0simple_conv_encoder/conv1d_6/conv1d/ExpandDims_1Ф
#simple_conv_encoder/conv1d_6/conv1dConv2D7simple_conv_encoder/conv1d_6/conv1d/ExpandDims:output:09simple_conv_encoder/conv1d_6/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ *
paddingSAME*
strides
2%
#simple_conv_encoder/conv1d_6/conv1dЖ
+simple_conv_encoder/conv1d_6/conv1d/SqueezeSqueeze,simple_conv_encoder/conv1d_6/conv1d:output:0*
T0*,
_output_shapes
:         ќ *
squeeze_dims

§        2-
+simple_conv_encoder/conv1d_6/conv1d/Squeezeс
3simple_conv_encoder/conv1d_6/BiasAdd/ReadVariableOpReadVariableOp<simple_conv_encoder_conv1d_6_biasadd_readvariableop_resource*
_output_shapes
: *
dtype025
3simple_conv_encoder/conv1d_6/BiasAdd/ReadVariableOpЂ
$simple_conv_encoder/conv1d_6/BiasAddBiasAdd4simple_conv_encoder/conv1d_6/conv1d/Squeeze:output:0;simple_conv_encoder/conv1d_6/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ 2&
$simple_conv_encoder/conv1d_6/BiasAdd┤
!simple_conv_encoder/conv1d_6/TanhTanh-simple_conv_encoder/conv1d_6/BiasAdd:output:0*
T0*,
_output_shapes
:         ќ 2#
!simple_conv_encoder/conv1d_6/Tanh│
2simple_conv_encoder/conv1d_7/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        24
2simple_conv_encoder/conv1d_7/conv1d/ExpandDims/dimЇ
.simple_conv_encoder/conv1d_7/conv1d/ExpandDims
ExpandDims%simple_conv_encoder/conv1d_6/Tanh:y:0;simple_conv_encoder/conv1d_7/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ 20
.simple_conv_encoder/conv1d_7/conv1d/ExpandDimsЈ
?simple_conv_encoder/conv1d_7/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpHsimple_conv_encoder_conv1d_7_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02A
?simple_conv_encoder/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp«
4simple_conv_encoder/conv1d_7/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4simple_conv_encoder/conv1d_7/conv1d/ExpandDims_1/dimФ
0simple_conv_encoder/conv1d_7/conv1d/ExpandDims_1
ExpandDimsGsimple_conv_encoder/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp:value:0=simple_conv_encoder/conv1d_7/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 22
0simple_conv_encoder/conv1d_7/conv1d/ExpandDims_1Ф
#simple_conv_encoder/conv1d_7/conv1dConv2D7simple_conv_encoder/conv1d_7/conv1d/ExpandDims:output:09simple_conv_encoder/conv1d_7/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ*
paddingSAME*
strides
2%
#simple_conv_encoder/conv1d_7/conv1dЖ
+simple_conv_encoder/conv1d_7/conv1d/SqueezeSqueeze,simple_conv_encoder/conv1d_7/conv1d:output:0*
T0*,
_output_shapes
:         ќ*
squeeze_dims

§        2-
+simple_conv_encoder/conv1d_7/conv1d/Squeezeс
3simple_conv_encoder/conv1d_7/BiasAdd/ReadVariableOpReadVariableOp<simple_conv_encoder_conv1d_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3simple_conv_encoder/conv1d_7/BiasAdd/ReadVariableOpЂ
$simple_conv_encoder/conv1d_7/BiasAddBiasAdd4simple_conv_encoder/conv1d_7/conv1d/Squeeze:output:0;simple_conv_encoder/conv1d_7/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ2&
$simple_conv_encoder/conv1d_7/BiasAdd┤
!simple_conv_encoder/conv1d_7/TanhTanh-simple_conv_encoder/conv1d_7/BiasAdd:output:0*
T0*,
_output_shapes
:         ќ2#
!simple_conv_encoder/conv1d_7/Tanh│
2simple_conv_encoder/conv1d_8/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        24
2simple_conv_encoder/conv1d_8/conv1d/ExpandDims/dimЇ
.simple_conv_encoder/conv1d_8/conv1d/ExpandDims
ExpandDims%simple_conv_encoder/conv1d_7/Tanh:y:0;simple_conv_encoder/conv1d_8/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ20
.simple_conv_encoder/conv1d_8/conv1d/ExpandDimsЈ
?simple_conv_encoder/conv1d_8/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpHsimple_conv_encoder_conv1d_8_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02A
?simple_conv_encoder/conv1d_8/conv1d/ExpandDims_1/ReadVariableOp«
4simple_conv_encoder/conv1d_8/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4simple_conv_encoder/conv1d_8/conv1d/ExpandDims_1/dimФ
0simple_conv_encoder/conv1d_8/conv1d/ExpandDims_1
ExpandDimsGsimple_conv_encoder/conv1d_8/conv1d/ExpandDims_1/ReadVariableOp:value:0=simple_conv_encoder/conv1d_8/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0simple_conv_encoder/conv1d_8/conv1d/ExpandDims_1Ф
#simple_conv_encoder/conv1d_8/conv1dConv2D7simple_conv_encoder/conv1d_8/conv1d/ExpandDims:output:09simple_conv_encoder/conv1d_8/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ*
paddingSAME*
strides
2%
#simple_conv_encoder/conv1d_8/conv1dЖ
+simple_conv_encoder/conv1d_8/conv1d/SqueezeSqueeze,simple_conv_encoder/conv1d_8/conv1d:output:0*
T0*,
_output_shapes
:         ќ*
squeeze_dims

§        2-
+simple_conv_encoder/conv1d_8/conv1d/Squeezeс
3simple_conv_encoder/conv1d_8/BiasAdd/ReadVariableOpReadVariableOp<simple_conv_encoder_conv1d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3simple_conv_encoder/conv1d_8/BiasAdd/ReadVariableOpЂ
$simple_conv_encoder/conv1d_8/BiasAddBiasAdd4simple_conv_encoder/conv1d_8/conv1d/Squeeze:output:0;simple_conv_encoder/conv1d_8/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ2&
$simple_conv_encoder/conv1d_8/BiasAdd┤
!simple_conv_encoder/conv1d_8/TanhTanh-simple_conv_encoder/conv1d_8/BiasAdd:output:0*
T0*,
_output_shapes
:         ќ2#
!simple_conv_encoder/conv1d_8/Tanh│
2simple_conv_encoder/conv1d_9/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        24
2simple_conv_encoder/conv1d_9/conv1d/ExpandDims/dimЇ
.simple_conv_encoder/conv1d_9/conv1d/ExpandDims
ExpandDims%simple_conv_encoder/conv1d_8/Tanh:y:0;simple_conv_encoder/conv1d_9/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ20
.simple_conv_encoder/conv1d_9/conv1d/ExpandDimsЈ
?simple_conv_encoder/conv1d_9/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpHsimple_conv_encoder_conv1d_9_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02A
?simple_conv_encoder/conv1d_9/conv1d/ExpandDims_1/ReadVariableOp«
4simple_conv_encoder/conv1d_9/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 26
4simple_conv_encoder/conv1d_9/conv1d/ExpandDims_1/dimФ
0simple_conv_encoder/conv1d_9/conv1d/ExpandDims_1
ExpandDimsGsimple_conv_encoder/conv1d_9/conv1d/ExpandDims_1/ReadVariableOp:value:0=simple_conv_encoder/conv1d_9/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:22
0simple_conv_encoder/conv1d_9/conv1d/ExpandDims_1Ф
#simple_conv_encoder/conv1d_9/conv1dConv2D7simple_conv_encoder/conv1d_9/conv1d/ExpandDims:output:09simple_conv_encoder/conv1d_9/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ*
paddingSAME*
strides
2%
#simple_conv_encoder/conv1d_9/conv1dЖ
+simple_conv_encoder/conv1d_9/conv1d/SqueezeSqueeze,simple_conv_encoder/conv1d_9/conv1d:output:0*
T0*,
_output_shapes
:         ќ*
squeeze_dims

§        2-
+simple_conv_encoder/conv1d_9/conv1d/Squeezeс
3simple_conv_encoder/conv1d_9/BiasAdd/ReadVariableOpReadVariableOp<simple_conv_encoder_conv1d_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype025
3simple_conv_encoder/conv1d_9/BiasAdd/ReadVariableOpЂ
$simple_conv_encoder/conv1d_9/BiasAddBiasAdd4simple_conv_encoder/conv1d_9/conv1d/Squeeze:output:0;simple_conv_encoder/conv1d_9/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ2&
$simple_conv_encoder/conv1d_9/BiasAdd┤
!simple_conv_encoder/conv1d_9/TanhTanh-simple_conv_encoder/conv1d_9/BiasAdd:output:0*
T0*,
_output_shapes
:         ќ2#
!simple_conv_encoder/conv1d_9/Tanhх
3simple_conv_encoder/conv1d_10/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        25
3simple_conv_encoder/conv1d_10/conv1d/ExpandDims/dimљ
/simple_conv_encoder/conv1d_10/conv1d/ExpandDims
ExpandDims%simple_conv_encoder/conv1d_9/Tanh:y:0<simple_conv_encoder/conv1d_10/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ21
/simple_conv_encoder/conv1d_10/conv1d/ExpandDimsњ
@simple_conv_encoder/conv1d_10/conv1d/ExpandDims_1/ReadVariableOpReadVariableOpIsimple_conv_encoder_conv1d_10_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02B
@simple_conv_encoder/conv1d_10/conv1d/ExpandDims_1/ReadVariableOp░
5simple_conv_encoder/conv1d_10/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 27
5simple_conv_encoder/conv1d_10/conv1d/ExpandDims_1/dim»
1simple_conv_encoder/conv1d_10/conv1d/ExpandDims_1
ExpandDimsHsimple_conv_encoder/conv1d_10/conv1d/ExpandDims_1/ReadVariableOp:value:0>simple_conv_encoder/conv1d_10/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:23
1simple_conv_encoder/conv1d_10/conv1d/ExpandDims_1»
$simple_conv_encoder/conv1d_10/conv1dConv2D8simple_conv_encoder/conv1d_10/conv1d/ExpandDims:output:0:simple_conv_encoder/conv1d_10/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ*
paddingSAME*
strides
2&
$simple_conv_encoder/conv1d_10/conv1dь
,simple_conv_encoder/conv1d_10/conv1d/SqueezeSqueeze-simple_conv_encoder/conv1d_10/conv1d:output:0*
T0*,
_output_shapes
:         ќ*
squeeze_dims

§        2.
,simple_conv_encoder/conv1d_10/conv1d/SqueezeТ
4simple_conv_encoder/conv1d_10/BiasAdd/ReadVariableOpReadVariableOp=simple_conv_encoder_conv1d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype026
4simple_conv_encoder/conv1d_10/BiasAdd/ReadVariableOpЁ
%simple_conv_encoder/conv1d_10/BiasAddBiasAdd5simple_conv_encoder/conv1d_10/conv1d/Squeeze:output:0<simple_conv_encoder/conv1d_10/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ2'
%simple_conv_encoder/conv1d_10/BiasAddи
"simple_conv_encoder/conv1d_10/TanhTanh.simple_conv_encoder/conv1d_10/BiasAdd:output:0*
T0*,
_output_shapes
:         ќ2$
"simple_conv_encoder/conv1d_10/TanhЌ
!simple_conv_encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    ,  2#
!simple_conv_encoder/flatten/Const▄
#simple_conv_encoder/flatten/ReshapeReshape&simple_conv_encoder/conv1d_10/Tanh:y:0*simple_conv_encoder/flatten/Const:output:0*
T0*(
_output_shapes
:         г2%
#simple_conv_encoder/flatten/Reshape▄
/simple_conv_encoder/dense/MatMul/ReadVariableOpReadVariableOp8simple_conv_encoder_dense_matmul_readvariableop_resource*
_output_shapes
:	г*
dtype021
/simple_conv_encoder/dense/MatMul/ReadVariableOpу
 simple_conv_encoder/dense/MatMulMatMul,simple_conv_encoder/flatten/Reshape:output:07simple_conv_encoder/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2"
 simple_conv_encoder/dense/MatMul┌
0simple_conv_encoder/dense/BiasAdd/ReadVariableOpReadVariableOp9simple_conv_encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0simple_conv_encoder/dense/BiasAdd/ReadVariableOpж
!simple_conv_encoder/dense/BiasAddBiasAdd*simple_conv_encoder/dense/MatMul:product:08simple_conv_encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2#
!simple_conv_encoder/dense/BiasAddд
simple_conv_encoder/dense/TanhTanh*simple_conv_encoder/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         2 
simple_conv_encoder/dense/TanhЂ
IdentityIdentity"simple_conv_encoder/dense/Tanh:y:02^simple_conv_encoder/conv1d/BiasAdd/ReadVariableOp>^simple_conv_encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOp4^simple_conv_encoder/conv1d_1/BiasAdd/ReadVariableOp@^simple_conv_encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp5^simple_conv_encoder/conv1d_10/BiasAdd/ReadVariableOpA^simple_conv_encoder/conv1d_10/conv1d/ExpandDims_1/ReadVariableOp4^simple_conv_encoder/conv1d_2/BiasAdd/ReadVariableOp@^simple_conv_encoder/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp4^simple_conv_encoder/conv1d_3/BiasAdd/ReadVariableOp@^simple_conv_encoder/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp4^simple_conv_encoder/conv1d_4/BiasAdd/ReadVariableOp@^simple_conv_encoder/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp4^simple_conv_encoder/conv1d_5/BiasAdd/ReadVariableOp@^simple_conv_encoder/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp4^simple_conv_encoder/conv1d_6/BiasAdd/ReadVariableOp@^simple_conv_encoder/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp4^simple_conv_encoder/conv1d_7/BiasAdd/ReadVariableOp@^simple_conv_encoder/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp4^simple_conv_encoder/conv1d_8/BiasAdd/ReadVariableOp@^simple_conv_encoder/conv1d_8/conv1d/ExpandDims_1/ReadVariableOp4^simple_conv_encoder/conv1d_9/BiasAdd/ReadVariableOp@^simple_conv_encoder/conv1d_9/conv1d/ExpandDims_1/ReadVariableOp1^simple_conv_encoder/dense/BiasAdd/ReadVariableOp0^simple_conv_encoder/dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*І
_input_shapesz
x:         ќ::::::::::::::::::::::::2f
1simple_conv_encoder/conv1d/BiasAdd/ReadVariableOp1simple_conv_encoder/conv1d/BiasAdd/ReadVariableOp2~
=simple_conv_encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOp=simple_conv_encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOp2j
3simple_conv_encoder/conv1d_1/BiasAdd/ReadVariableOp3simple_conv_encoder/conv1d_1/BiasAdd/ReadVariableOp2ѓ
?simple_conv_encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp?simple_conv_encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp2l
4simple_conv_encoder/conv1d_10/BiasAdd/ReadVariableOp4simple_conv_encoder/conv1d_10/BiasAdd/ReadVariableOp2ё
@simple_conv_encoder/conv1d_10/conv1d/ExpandDims_1/ReadVariableOp@simple_conv_encoder/conv1d_10/conv1d/ExpandDims_1/ReadVariableOp2j
3simple_conv_encoder/conv1d_2/BiasAdd/ReadVariableOp3simple_conv_encoder/conv1d_2/BiasAdd/ReadVariableOp2ѓ
?simple_conv_encoder/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp?simple_conv_encoder/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp2j
3simple_conv_encoder/conv1d_3/BiasAdd/ReadVariableOp3simple_conv_encoder/conv1d_3/BiasAdd/ReadVariableOp2ѓ
?simple_conv_encoder/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp?simple_conv_encoder/conv1d_3/conv1d/ExpandDims_1/ReadVariableOp2j
3simple_conv_encoder/conv1d_4/BiasAdd/ReadVariableOp3simple_conv_encoder/conv1d_4/BiasAdd/ReadVariableOp2ѓ
?simple_conv_encoder/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp?simple_conv_encoder/conv1d_4/conv1d/ExpandDims_1/ReadVariableOp2j
3simple_conv_encoder/conv1d_5/BiasAdd/ReadVariableOp3simple_conv_encoder/conv1d_5/BiasAdd/ReadVariableOp2ѓ
?simple_conv_encoder/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp?simple_conv_encoder/conv1d_5/conv1d/ExpandDims_1/ReadVariableOp2j
3simple_conv_encoder/conv1d_6/BiasAdd/ReadVariableOp3simple_conv_encoder/conv1d_6/BiasAdd/ReadVariableOp2ѓ
?simple_conv_encoder/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp?simple_conv_encoder/conv1d_6/conv1d/ExpandDims_1/ReadVariableOp2j
3simple_conv_encoder/conv1d_7/BiasAdd/ReadVariableOp3simple_conv_encoder/conv1d_7/BiasAdd/ReadVariableOp2ѓ
?simple_conv_encoder/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp?simple_conv_encoder/conv1d_7/conv1d/ExpandDims_1/ReadVariableOp2j
3simple_conv_encoder/conv1d_8/BiasAdd/ReadVariableOp3simple_conv_encoder/conv1d_8/BiasAdd/ReadVariableOp2ѓ
?simple_conv_encoder/conv1d_8/conv1d/ExpandDims_1/ReadVariableOp?simple_conv_encoder/conv1d_8/conv1d/ExpandDims_1/ReadVariableOp2j
3simple_conv_encoder/conv1d_9/BiasAdd/ReadVariableOp3simple_conv_encoder/conv1d_9/BiasAdd/ReadVariableOp2ѓ
?simple_conv_encoder/conv1d_9/conv1d/ExpandDims_1/ReadVariableOp?simple_conv_encoder/conv1d_9/conv1d/ExpandDims_1/ReadVariableOp2d
0simple_conv_encoder/dense/BiasAdd/ReadVariableOp0simple_conv_encoder/dense/BiasAdd/ReadVariableOp2b
/simple_conv_encoder/dense/MatMul/ReadVariableOp/simple_conv_encoder/dense/MatMul/ReadVariableOp:U Q
,
_output_shapes
:         ќ
!
_user_specified_name	input_1
џ
C
'__inference_flatten_layer_call_fn_13164

inputs
identity┴
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         г* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_130082
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         г2

Identity"
identityIdentity:output:0*+
_input_shapes
:         ќ:T P
,
_output_shapes
:         ќ
 
_user_specified_nameinputs
Ј
Ш
C__inference_conv1d_9_layer_call_and_return_conditional_losses_12954

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ќ*
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ќ2
TanhЪ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:         ќ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ќ
 
_user_specified_nameinputs
Ь
}
(__inference_conv1d_2_layer_call_fn_13259

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ќ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_127302
StatefulPartitionedCallЊ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         ќ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ@::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ќ@
 
_user_specified_nameinputs
┘
Я
3__inference_simple_conv_encoder_layer_call_fn_13098
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identityѕбStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *W
fRRP
N__inference_simple_conv_encoder_layer_call_and_return_conditional_losses_130442
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*І
_input_shapesz
x:         ќ::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:         ќ
!
_user_specified_name	input_1
Ь
}
(__inference_conv1d_4_layer_call_fn_13309

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ќ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_conv1d_4_layer_call_and_return_conditional_losses_127942
StatefulPartitionedCallЊ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         ќ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ@::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ќ@
 
_user_specified_nameinputs
Ј
Ш
C__inference_conv1d_2_layer_call_and_return_conditional_losses_12730

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ@*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ќ@*
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ@2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ќ@2
TanhЪ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:         ќ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ќ@
 
_user_specified_nameinputs
Ј
Ш
C__inference_conv1d_5_layer_call_and_return_conditional_losses_12826

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ@*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ќ@*
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ@2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ќ@2
TanhЪ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:         ќ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ќ@
 
_user_specified_nameinputs
љ
э
D__inference_conv1d_10_layer_call_and_return_conditional_losses_12986

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ќ*
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ќ2
TanhЪ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:         ќ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ќ
 
_user_specified_nameinputs
Ј
Ш
C__inference_conv1d_7_layer_call_and_return_conditional_losses_13375

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ 2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ќ*
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ќ2
TanhЪ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:         ќ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ќ 
 
_user_specified_nameinputs
Ь
}
(__inference_conv1d_5_layer_call_fn_13334

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ќ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_conv1d_5_layer_call_and_return_conditional_losses_128262
StatefulPartitionedCallЊ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         ќ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ@::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ќ@
 
_user_specified_nameinputs
Ј
Ш
C__inference_conv1d_5_layer_call_and_return_conditional_losses_13325

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ@*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ќ@*
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ@2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ќ@2
TanhЪ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:         ќ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ќ@
 
_user_specified_nameinputs
Ј
Ш
C__inference_conv1d_3_layer_call_and_return_conditional_losses_12762

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ@*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ќ@*
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ@2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ќ@2
TanhЪ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:         ќ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ќ@
 
_user_specified_nameinputs
Ь
}
(__inference_conv1d_9_layer_call_fn_13434

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ќ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_conv1d_9_layer_call_and_return_conditional_losses_129542
StatefulPartitionedCallЊ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         ќ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ќ
 
_user_specified_nameinputs
Ь
}
(__inference_conv1d_3_layer_call_fn_13284

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ќ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_conv1d_3_layer_call_and_return_conditional_losses_127622
StatefulPartitionedCallЊ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         ќ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ@::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ќ@
 
_user_specified_nameinputs
╠@
Т
N__inference_simple_conv_encoder_layer_call_and_return_conditional_losses_13044
input_1
conv1d_12677
conv1d_12679
conv1d_1_12709
conv1d_1_12711
conv1d_2_12741
conv1d_2_12743
conv1d_3_12773
conv1d_3_12775
conv1d_4_12805
conv1d_4_12807
conv1d_5_12837
conv1d_5_12839
conv1d_6_12869
conv1d_6_12871
conv1d_7_12901
conv1d_7_12903
conv1d_8_12933
conv1d_8_12935
conv1d_9_12965
conv1d_9_12967
conv1d_10_12997
conv1d_10_12999
dense_13038
dense_13040
identityѕбconv1d/StatefulPartitionedCallб conv1d_1/StatefulPartitionedCallб!conv1d_10/StatefulPartitionedCallб conv1d_2/StatefulPartitionedCallб conv1d_3/StatefulPartitionedCallб conv1d_4/StatefulPartitionedCallб conv1d_5/StatefulPartitionedCallб conv1d_6/StatefulPartitionedCallб conv1d_7/StatefulPartitionedCallб conv1d_8/StatefulPartitionedCallб conv1d_9/StatefulPartitionedCallбdense/StatefulPartitionedCallЇ
conv1d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1d_12677conv1d_12679*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ќ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_126662 
conv1d/StatefulPartitionedCallи
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0conv1d_1_12709conv1d_1_12711*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ќ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_126982"
 conv1d_1/StatefulPartitionedCall╣
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0conv1d_2_12741conv1d_2_12743*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ќ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_conv1d_2_layer_call_and_return_conditional_losses_127302"
 conv1d_2/StatefulPartitionedCall╣
 conv1d_3/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0conv1d_3_12773conv1d_3_12775*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ќ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_conv1d_3_layer_call_and_return_conditional_losses_127622"
 conv1d_3/StatefulPartitionedCall╣
 conv1d_4/StatefulPartitionedCallStatefulPartitionedCall)conv1d_3/StatefulPartitionedCall:output:0conv1d_4_12805conv1d_4_12807*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ќ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_conv1d_4_layer_call_and_return_conditional_losses_127942"
 conv1d_4/StatefulPartitionedCall╣
 conv1d_5/StatefulPartitionedCallStatefulPartitionedCall)conv1d_4/StatefulPartitionedCall:output:0conv1d_5_12837conv1d_5_12839*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ќ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_conv1d_5_layer_call_and_return_conditional_losses_128262"
 conv1d_5/StatefulPartitionedCall╣
 conv1d_6/StatefulPartitionedCallStatefulPartitionedCall)conv1d_5/StatefulPartitionedCall:output:0conv1d_6_12869conv1d_6_12871*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ќ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_conv1d_6_layer_call_and_return_conditional_losses_128582"
 conv1d_6/StatefulPartitionedCall╣
 conv1d_7/StatefulPartitionedCallStatefulPartitionedCall)conv1d_6/StatefulPartitionedCall:output:0conv1d_7_12901conv1d_7_12903*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ќ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_conv1d_7_layer_call_and_return_conditional_losses_128902"
 conv1d_7/StatefulPartitionedCall╣
 conv1d_8/StatefulPartitionedCallStatefulPartitionedCall)conv1d_7/StatefulPartitionedCall:output:0conv1d_8_12933conv1d_8_12935*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ќ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_conv1d_8_layer_call_and_return_conditional_losses_129222"
 conv1d_8/StatefulPartitionedCall╣
 conv1d_9/StatefulPartitionedCallStatefulPartitionedCall)conv1d_8/StatefulPartitionedCall:output:0conv1d_9_12965conv1d_9_12967*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ќ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_conv1d_9_layer_call_and_return_conditional_losses_129542"
 conv1d_9/StatefulPartitionedCallЙ
!conv1d_10/StatefulPartitionedCallStatefulPartitionedCall)conv1d_9/StatefulPartitionedCall:output:0conv1d_10_12997conv1d_10_12999*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ќ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv1d_10_layer_call_and_return_conditional_losses_129862#
!conv1d_10/StatefulPartitionedCallш
flatten/PartitionedCallPartitionedCall*conv1d_10/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         г* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_130082
flatten/PartitionedCallю
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_13038dense_13040*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_130272
dense/StatefulPartitionedCallџ
IdentityIdentity&dense/StatefulPartitionedCall:output:0^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall"^conv1d_10/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall!^conv1d_3/StatefulPartitionedCall!^conv1d_4/StatefulPartitionedCall!^conv1d_5/StatefulPartitionedCall!^conv1d_6/StatefulPartitionedCall!^conv1d_7/StatefulPartitionedCall!^conv1d_8/StatefulPartitionedCall!^conv1d_9/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*І
_input_shapesz
x:         ќ::::::::::::::::::::::::2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2F
!conv1d_10/StatefulPartitionedCall!conv1d_10/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2D
 conv1d_3/StatefulPartitionedCall conv1d_3/StatefulPartitionedCall2D
 conv1d_4/StatefulPartitionedCall conv1d_4/StatefulPartitionedCall2D
 conv1d_5/StatefulPartitionedCall conv1d_5/StatefulPartitionedCall2D
 conv1d_6/StatefulPartitionedCall conv1d_6/StatefulPartitionedCall2D
 conv1d_7/StatefulPartitionedCall conv1d_7/StatefulPartitionedCall2D
 conv1d_8/StatefulPartitionedCall conv1d_8/StatefulPartitionedCall2D
 conv1d_9/StatefulPartitionedCall conv1d_9/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:U Q
,
_output_shapes
:         ќ
!
_user_specified_name	input_1
Ї
З
A__inference_conv1d_layer_call_and_return_conditional_losses_12666

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ@*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ќ@*
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ@2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ќ@2
TanhЪ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:         ќ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ќ
 
_user_specified_nameinputs
Ј
Ш
C__inference_conv1d_2_layer_call_and_return_conditional_losses_13250

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ@*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ќ@*
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ@2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ќ@2
TanhЪ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:         ќ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ќ@
 
_user_specified_nameinputs
с	
┘
@__inference_dense_layer_call_and_return_conditional_losses_13175

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	г*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddX
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:         2
TanhЇ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         г::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         г
 
_user_specified_nameinputs
Џ
л
#__inference_signature_wrapper_13153
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20

unknown_21

unknown_22
identityѕбStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22*$
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *:
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8ѓ *)
f$R"
 __inference__wrapped_model_126462
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*І
_input_shapesz
x:         ќ::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:         ќ
!
_user_specified_name	input_1
Ј
Ш
C__inference_conv1d_1_layer_call_and_return_conditional_losses_12698

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ@*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ќ@*
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ@2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ќ@2
TanhЪ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:         ќ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ќ@
 
_user_specified_nameinputs
о
z
%__inference_dense_layer_call_fn_13184

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_130272
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         г::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         г
 
_user_specified_nameinputs
Ј
Ш
C__inference_conv1d_6_layer_call_and_return_conditional_losses_12858

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ *
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ќ *
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ 2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ќ 2
TanhЪ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:         ќ 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ќ@
 
_user_specified_nameinputs
┤
^
B__inference_flatten_layer_call_and_return_conditional_losses_13008

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"    ,  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:         г2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:         г2

Identity"
identityIdentity:output:0*+
_input_shapes
:         ќ:T P
,
_output_shapes
:         ќ
 
_user_specified_nameinputs
Ј
Ш
C__inference_conv1d_4_layer_call_and_return_conditional_losses_13300

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ@*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ќ@*
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ@2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ќ@2
TanhЪ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:         ќ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ќ@
 
_user_specified_nameinputs
р=
Ф
__inference__traced_save_13554
file_prefix?
;savev2_simple_conv_encoder_dense_kernel_read_readvariableop=
9savev2_simple_conv_encoder_dense_bias_read_readvariableop@
<savev2_simple_conv_encoder_conv1d_kernel_read_readvariableop>
:savev2_simple_conv_encoder_conv1d_bias_read_readvariableopB
>savev2_simple_conv_encoder_conv1d_1_kernel_read_readvariableop@
<savev2_simple_conv_encoder_conv1d_1_bias_read_readvariableopB
>savev2_simple_conv_encoder_conv1d_2_kernel_read_readvariableop@
<savev2_simple_conv_encoder_conv1d_2_bias_read_readvariableopB
>savev2_simple_conv_encoder_conv1d_3_kernel_read_readvariableop@
<savev2_simple_conv_encoder_conv1d_3_bias_read_readvariableopB
>savev2_simple_conv_encoder_conv1d_4_kernel_read_readvariableop@
<savev2_simple_conv_encoder_conv1d_4_bias_read_readvariableopB
>savev2_simple_conv_encoder_conv1d_5_kernel_read_readvariableop@
<savev2_simple_conv_encoder_conv1d_5_bias_read_readvariableopB
>savev2_simple_conv_encoder_conv1d_6_kernel_read_readvariableop@
<savev2_simple_conv_encoder_conv1d_6_bias_read_readvariableopB
>savev2_simple_conv_encoder_conv1d_7_kernel_read_readvariableop@
<savev2_simple_conv_encoder_conv1d_7_bias_read_readvariableopB
>savev2_simple_conv_encoder_conv1d_8_kernel_read_readvariableop@
<savev2_simple_conv_encoder_conv1d_8_bias_read_readvariableopB
>savev2_simple_conv_encoder_conv1d_9_kernel_read_readvariableop@
<savev2_simple_conv_encoder_conv1d_9_bias_read_readvariableopC
?savev2_simple_conv_encoder_conv1d_10_kernel_read_readvariableopA
=savev2_simple_conv_encoder_conv1d_10_bias_read_readvariableop
savev2_const

identity_1ѕбMergeV2CheckpointsЈ
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename┼

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*О	
value═	B╩	B%out/kernel/.ATTRIBUTES/VARIABLE_VALUEB#out/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names║
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices▓
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0;savev2_simple_conv_encoder_dense_kernel_read_readvariableop9savev2_simple_conv_encoder_dense_bias_read_readvariableop<savev2_simple_conv_encoder_conv1d_kernel_read_readvariableop:savev2_simple_conv_encoder_conv1d_bias_read_readvariableop>savev2_simple_conv_encoder_conv1d_1_kernel_read_readvariableop<savev2_simple_conv_encoder_conv1d_1_bias_read_readvariableop>savev2_simple_conv_encoder_conv1d_2_kernel_read_readvariableop<savev2_simple_conv_encoder_conv1d_2_bias_read_readvariableop>savev2_simple_conv_encoder_conv1d_3_kernel_read_readvariableop<savev2_simple_conv_encoder_conv1d_3_bias_read_readvariableop>savev2_simple_conv_encoder_conv1d_4_kernel_read_readvariableop<savev2_simple_conv_encoder_conv1d_4_bias_read_readvariableop>savev2_simple_conv_encoder_conv1d_5_kernel_read_readvariableop<savev2_simple_conv_encoder_conv1d_5_bias_read_readvariableop>savev2_simple_conv_encoder_conv1d_6_kernel_read_readvariableop<savev2_simple_conv_encoder_conv1d_6_bias_read_readvariableop>savev2_simple_conv_encoder_conv1d_7_kernel_read_readvariableop<savev2_simple_conv_encoder_conv1d_7_bias_read_readvariableop>savev2_simple_conv_encoder_conv1d_8_kernel_read_readvariableop<savev2_simple_conv_encoder_conv1d_8_bias_read_readvariableop>savev2_simple_conv_encoder_conv1d_9_kernel_read_readvariableop<savev2_simple_conv_encoder_conv1d_9_bias_read_readvariableop?savev2_simple_conv_encoder_conv1d_10_kernel_read_readvariableop=savev2_simple_conv_encoder_conv1d_10_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
22
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesА
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*є
_input_shapesЗ
ы: :	г::@:@:@@:@:@@:@:@@:@:@@:@:@@:@:@ : : :::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	г: 

_output_shapes
::($
"
_output_shapes
:@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:(	$
"
_output_shapes
:@@: 


_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:($
"
_output_shapes
:@@: 

_output_shapes
:@:($
"
_output_shapes
:@ : 

_output_shapes
: :($
"
_output_shapes
: : 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
Ј
Ш
C__inference_conv1d_6_layer_call_and_return_conditional_losses_13350

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@ *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@ 2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ *
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ќ *
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ 2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ќ 2
TanhЪ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:         ќ 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ќ@
 
_user_specified_nameinputs
Ь
}
(__inference_conv1d_8_layer_call_fn_13409

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ќ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_conv1d_8_layer_call_and_return_conditional_losses_129222
StatefulPartitionedCallЊ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         ќ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ќ
 
_user_specified_nameinputs
Ј
Ш
C__inference_conv1d_8_layer_call_and_return_conditional_losses_12922

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ќ*
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ќ2
TanhЪ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:         ќ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ќ
 
_user_specified_nameinputs
Ь
}
(__inference_conv1d_1_layer_call_fn_13234

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ќ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_conv1d_1_layer_call_and_return_conditional_losses_126982
StatefulPartitionedCallЊ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         ќ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ@::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ќ@
 
_user_specified_nameinputs
Ь
}
(__inference_conv1d_7_layer_call_fn_13384

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ќ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_conv1d_7_layer_call_and_return_conditional_losses_128902
StatefulPartitionedCallЊ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         ќ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ќ 
 
_user_specified_nameinputs
­
~
)__inference_conv1d_10_layer_call_fn_13459

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ќ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_conv1d_10_layer_call_and_return_conditional_losses_129862
StatefulPartitionedCallЊ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         ќ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ќ
 
_user_specified_nameinputs
Ј
Ш
C__inference_conv1d_9_layer_call_and_return_conditional_losses_13425

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ќ*
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ќ2
TanhЪ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:         ќ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ќ
 
_user_specified_nameinputs
Ј
Ш
C__inference_conv1d_3_layer_call_and_return_conditional_losses_13275

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ@*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ќ@*
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ@2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ќ@2
TanhЪ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:         ќ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ќ@
 
_user_specified_nameinputs
Ь
}
(__inference_conv1d_6_layer_call_fn_13359

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ќ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_conv1d_6_layer_call_and_return_conditional_losses_128582
StatefulPartitionedCallЊ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         ќ 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ@::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ќ@
 
_user_specified_nameinputs
╬j
п
!__inference__traced_restore_13636
file_prefix5
1assignvariableop_simple_conv_encoder_dense_kernel5
1assignvariableop_1_simple_conv_encoder_dense_bias8
4assignvariableop_2_simple_conv_encoder_conv1d_kernel6
2assignvariableop_3_simple_conv_encoder_conv1d_bias:
6assignvariableop_4_simple_conv_encoder_conv1d_1_kernel8
4assignvariableop_5_simple_conv_encoder_conv1d_1_bias:
6assignvariableop_6_simple_conv_encoder_conv1d_2_kernel8
4assignvariableop_7_simple_conv_encoder_conv1d_2_bias:
6assignvariableop_8_simple_conv_encoder_conv1d_3_kernel8
4assignvariableop_9_simple_conv_encoder_conv1d_3_bias;
7assignvariableop_10_simple_conv_encoder_conv1d_4_kernel9
5assignvariableop_11_simple_conv_encoder_conv1d_4_bias;
7assignvariableop_12_simple_conv_encoder_conv1d_5_kernel9
5assignvariableop_13_simple_conv_encoder_conv1d_5_bias;
7assignvariableop_14_simple_conv_encoder_conv1d_6_kernel9
5assignvariableop_15_simple_conv_encoder_conv1d_6_bias;
7assignvariableop_16_simple_conv_encoder_conv1d_7_kernel9
5assignvariableop_17_simple_conv_encoder_conv1d_7_bias;
7assignvariableop_18_simple_conv_encoder_conv1d_8_kernel9
5assignvariableop_19_simple_conv_encoder_conv1d_8_bias;
7assignvariableop_20_simple_conv_encoder_conv1d_9_kernel9
5assignvariableop_21_simple_conv_encoder_conv1d_9_bias<
8assignvariableop_22_simple_conv_encoder_conv1d_10_kernel:
6assignvariableop_23_simple_conv_encoder_conv1d_10_bias
identity_25ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9╦

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*О	
value═	B╩	B%out/kernel/.ATTRIBUTES/VARIABLE_VALUEB#out/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/19/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/20/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/21/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names└
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesе
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity░
AssignVariableOpAssignVariableOp1assignvariableop_simple_conv_encoder_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Х
AssignVariableOp_1AssignVariableOp1assignvariableop_1_simple_conv_encoder_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2╣
AssignVariableOp_2AssignVariableOp4assignvariableop_2_simple_conv_encoder_conv1d_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3и
AssignVariableOp_3AssignVariableOp2assignvariableop_3_simple_conv_encoder_conv1d_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4╗
AssignVariableOp_4AssignVariableOp6assignvariableop_4_simple_conv_encoder_conv1d_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5╣
AssignVariableOp_5AssignVariableOp4assignvariableop_5_simple_conv_encoder_conv1d_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6╗
AssignVariableOp_6AssignVariableOp6assignvariableop_6_simple_conv_encoder_conv1d_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7╣
AssignVariableOp_7AssignVariableOp4assignvariableop_7_simple_conv_encoder_conv1d_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8╗
AssignVariableOp_8AssignVariableOp6assignvariableop_8_simple_conv_encoder_conv1d_3_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9╣
AssignVariableOp_9AssignVariableOp4assignvariableop_9_simple_conv_encoder_conv1d_3_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10┐
AssignVariableOp_10AssignVariableOp7assignvariableop_10_simple_conv_encoder_conv1d_4_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11й
AssignVariableOp_11AssignVariableOp5assignvariableop_11_simple_conv_encoder_conv1d_4_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12┐
AssignVariableOp_12AssignVariableOp7assignvariableop_12_simple_conv_encoder_conv1d_5_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13й
AssignVariableOp_13AssignVariableOp5assignvariableop_13_simple_conv_encoder_conv1d_5_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14┐
AssignVariableOp_14AssignVariableOp7assignvariableop_14_simple_conv_encoder_conv1d_6_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15й
AssignVariableOp_15AssignVariableOp5assignvariableop_15_simple_conv_encoder_conv1d_6_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16┐
AssignVariableOp_16AssignVariableOp7assignvariableop_16_simple_conv_encoder_conv1d_7_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17й
AssignVariableOp_17AssignVariableOp5assignvariableop_17_simple_conv_encoder_conv1d_7_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18┐
AssignVariableOp_18AssignVariableOp7assignvariableop_18_simple_conv_encoder_conv1d_8_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19й
AssignVariableOp_19AssignVariableOp5assignvariableop_19_simple_conv_encoder_conv1d_8_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20┐
AssignVariableOp_20AssignVariableOp7assignvariableop_20_simple_conv_encoder_conv1d_9_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21й
AssignVariableOp_21AssignVariableOp5assignvariableop_21_simple_conv_encoder_conv1d_9_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22└
AssignVariableOp_22AssignVariableOp8assignvariableop_22_simple_conv_encoder_conv1d_10_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23Й
AssignVariableOp_23AssignVariableOp6assignvariableop_23_simple_conv_encoder_conv1d_10_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_239
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЬ
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_24р
Identity_25IdentityIdentity_24:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_25"#
identity_25Identity_25:output:0*u
_input_shapesd
b: ::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ј
Ш
C__inference_conv1d_7_layer_call_and_return_conditional_losses_12890

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ 2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ќ*
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ќ2
TanhЪ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:         ќ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ќ 
 
_user_specified_nameinputs
Ј
Ш
C__inference_conv1d_4_layer_call_and_return_conditional_losses_12794

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpб"conv1d/ExpandDims_1/ReadVariableOpy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
§        2
conv1d/ExpandDims/dimЌ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ќ@2
conv1d/ExpandDimsИ
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dimи
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d/ExpandDims_1и
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ќ@*
paddingSAME*
strides
2
conv1dЊ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ќ@*
squeeze_dims

§        2
conv1d/Squeezeї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpЇ
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ќ@2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ќ@2
TanhЪ
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp#^conv1d/ExpandDims_1/ReadVariableOp*
T0*,
_output_shapes
:         ќ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2H
"conv1d/ExpandDims_1/ReadVariableOp"conv1d/ExpandDims_1/ReadVariableOp:T P
,
_output_shapes
:         ќ@
 
_user_specified_nameinputs
Ж
{
&__inference_conv1d_layer_call_fn_13209

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ќ@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *J
fERC
A__inference_conv1d_layer_call_and_return_conditional_losses_126662
StatefulPartitionedCallЊ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         ќ@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ќ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ќ
 
_user_specified_nameinputs"▒L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*░
serving_defaultю
@
input_15
serving_default_input_1:0         ќ<
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:рЕ
э
	convs
flatten
out
regularization_losses
trainable_variables
	variables
	keras_api

signatures
д_default_save_signature
+Д&call_and_return_all_conditional_losses
е__call__"Ќ
_tf_keras_model§{"class_name": "SimpleConvEncoder", "name": "simple_conv_encoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "SimpleConvEncoder"}}
n
	0

1
2
3
4
5
6
7
8
9
10"
trackable_list_wrapper
С
regularization_losses
trainable_variables
	variables
	keras_api
+Е&call_and_return_all_conditional_losses
ф__call__"М
_tf_keras_layer╣{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
№

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+Ф&call_and_return_all_conditional_losses
г__call__"╚
_tf_keras_layer«{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 300}}}, "build_input_shape": {"class_name": "TensorShape", "items": [282, 300]}}
 "
trackable_list_wrapper
о
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10
)11
*12
+13
,14
-15
.16
/17
018
119
220
321
22
23"
trackable_list_wrapper
о
0
1
 2
!3
"4
#5
$6
%7
&8
'9
(10
)11
*12
+13
,14
-15
.16
/17
018
119
220
321
22
23"
trackable_list_wrapper
╬
4metrics
regularization_losses
5layer_regularization_losses
trainable_variables
6layer_metrics

7layers
8non_trainable_variables
	variables
е__call__
д_default_save_signature
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
-
Гserving_default"
signature_map
р	

kernel
bias
9regularization_losses
:trainable_variables
;	variables
<	keras_api
+«&call_and_return_all_conditional_losses
»__call__"║
_tf_keras_layerа{"class_name": "Conv1D", "name": "conv1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [282, 150, 1]}}
у	

 kernel
!bias
=regularization_losses
>trainable_variables
?	variables
@	keras_api
+░&call_and_return_all_conditional_losses
▒__call__"└
_tf_keras_layerд{"class_name": "Conv1D", "name": "conv1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [282, 150, 64]}}
у	

"kernel
#bias
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
+▓&call_and_return_all_conditional_losses
│__call__"└
_tf_keras_layerд{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [282, 150, 64]}}
у	

$kernel
%bias
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
+┤&call_and_return_all_conditional_losses
х__call__"└
_tf_keras_layerд{"class_name": "Conv1D", "name": "conv1d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [282, 150, 64]}}
у	

&kernel
'bias
Iregularization_losses
Jtrainable_variables
K	variables
L	keras_api
+Х&call_and_return_all_conditional_losses
и__call__"└
_tf_keras_layerд{"class_name": "Conv1D", "name": "conv1d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [282, 150, 64]}}
у	

(kernel
)bias
Mregularization_losses
Ntrainable_variables
O	variables
P	keras_api
+И&call_and_return_all_conditional_losses
╣__call__"└
_tf_keras_layerд{"class_name": "Conv1D", "name": "conv1d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [282, 150, 64]}}
у	

*kernel
+bias
Qregularization_losses
Rtrainable_variables
S	variables
T	keras_api
+║&call_and_return_all_conditional_losses
╗__call__"└
_tf_keras_layerд{"class_name": "Conv1D", "name": "conv1d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_6", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [282, 150, 64]}}
у	

,kernel
-bias
Uregularization_losses
Vtrainable_variables
W	variables
X	keras_api
+╝&call_and_return_all_conditional_losses
й__call__"└
_tf_keras_layerд{"class_name": "Conv1D", "name": "conv1d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_7", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [282, 150, 32]}}
Т	

.kernel
/bias
Yregularization_losses
Ztrainable_variables
[	variables
\	keras_api
+Й&call_and_return_all_conditional_losses
┐__call__"┐
_tf_keras_layerЦ{"class_name": "Conv1D", "name": "conv1d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_8", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [282, 150, 16]}}
С	

0kernel
1bias
]regularization_losses
^trainable_variables
_	variables
`	keras_api
+└&call_and_return_all_conditional_losses
┴__call__"й
_tf_keras_layerБ{"class_name": "Conv1D", "name": "conv1d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_9", "trainable": true, "dtype": "float32", "filters": 4, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [282, 150, 8]}}
Т	

2kernel
3bias
aregularization_losses
btrainable_variables
c	variables
d	keras_api
+┬&call_and_return_all_conditional_losses
├__call__"┐
_tf_keras_layerЦ{"class_name": "Conv1D", "name": "conv1d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_10", "trainable": true, "dtype": "float32", "filters": 2, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [282, 150, 4]}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
emetrics
regularization_losses
flayer_regularization_losses
trainable_variables
glayer_metrics

hlayers
inon_trainable_variables
	variables
ф__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
3:1	г2 simple_conv_encoder/dense/kernel
,:*2simple_conv_encoder/dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
jmetrics
regularization_losses
klayer_regularization_losses
trainable_variables
llayer_metrics

mlayers
nnon_trainable_variables
	variables
г__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
7:5@2!simple_conv_encoder/conv1d/kernel
-:+@2simple_conv_encoder/conv1d/bias
9:7@@2#simple_conv_encoder/conv1d_1/kernel
/:-@2!simple_conv_encoder/conv1d_1/bias
9:7@@2#simple_conv_encoder/conv1d_2/kernel
/:-@2!simple_conv_encoder/conv1d_2/bias
9:7@@2#simple_conv_encoder/conv1d_3/kernel
/:-@2!simple_conv_encoder/conv1d_3/bias
9:7@@2#simple_conv_encoder/conv1d_4/kernel
/:-@2!simple_conv_encoder/conv1d_4/bias
9:7@@2#simple_conv_encoder/conv1d_5/kernel
/:-@2!simple_conv_encoder/conv1d_5/bias
9:7@ 2#simple_conv_encoder/conv1d_6/kernel
/:- 2!simple_conv_encoder/conv1d_6/bias
9:7 2#simple_conv_encoder/conv1d_7/kernel
/:-2!simple_conv_encoder/conv1d_7/bias
9:72#simple_conv_encoder/conv1d_8/kernel
/:-2!simple_conv_encoder/conv1d_8/bias
9:72#simple_conv_encoder/conv1d_9/kernel
/:-2!simple_conv_encoder/conv1d_9/bias
::82$simple_conv_encoder/conv1d_10/kernel
0:.2"simple_conv_encoder/conv1d_10/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
~
	0

1
2
3
4
5
6
7
8
9
10
11
12"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
░
ometrics
9regularization_losses
player_regularization_losses
:trainable_variables
qlayer_metrics

rlayers
snon_trainable_variables
;	variables
»__call__
+«&call_and_return_all_conditional_losses
'«"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
░
tmetrics
=regularization_losses
ulayer_regularization_losses
>trainable_variables
vlayer_metrics

wlayers
xnon_trainable_variables
?	variables
▒__call__
+░&call_and_return_all_conditional_losses
'░"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
░
ymetrics
Aregularization_losses
zlayer_regularization_losses
Btrainable_variables
{layer_metrics

|layers
}non_trainable_variables
C	variables
│__call__
+▓&call_and_return_all_conditional_losses
'▓"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
│
~metrics
Eregularization_losses
layer_regularization_losses
Ftrainable_variables
ђlayer_metrics
Ђlayers
ѓnon_trainable_variables
G	variables
х__call__
+┤&call_and_return_all_conditional_losses
'┤"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
х
Ѓmetrics
Iregularization_losses
 ёlayer_regularization_losses
Jtrainable_variables
Ёlayer_metrics
єlayers
Єnon_trainable_variables
K	variables
и__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
х
ѕmetrics
Mregularization_losses
 Ѕlayer_regularization_losses
Ntrainable_variables
іlayer_metrics
Іlayers
їnon_trainable_variables
O	variables
╣__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
х
Їmetrics
Qregularization_losses
 јlayer_regularization_losses
Rtrainable_variables
Јlayer_metrics
љlayers
Љnon_trainable_variables
S	variables
╗__call__
+║&call_and_return_all_conditional_losses
'║"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
х
њmetrics
Uregularization_losses
 Њlayer_regularization_losses
Vtrainable_variables
ћlayer_metrics
Ћlayers
ќnon_trainable_variables
W	variables
й__call__
+╝&call_and_return_all_conditional_losses
'╝"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
х
Ќmetrics
Yregularization_losses
 ўlayer_regularization_losses
Ztrainable_variables
Ўlayer_metrics
џlayers
Џnon_trainable_variables
[	variables
┐__call__
+Й&call_and_return_all_conditional_losses
'Й"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
х
юmetrics
]regularization_losses
 Юlayer_regularization_losses
^trainable_variables
ъlayer_metrics
Ъlayers
аnon_trainable_variables
_	variables
┴__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
х
Аmetrics
aregularization_losses
 бlayer_regularization_losses
btrainable_variables
Бlayer_metrics
цlayers
Цnon_trainable_variables
c	variables
├__call__
+┬&call_and_return_all_conditional_losses
'┬"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
с2Я
 __inference__wrapped_model_12646╗
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *+б(
&і#
input_1         ќ
А2ъ
N__inference_simple_conv_encoder_layer_call_and_return_conditional_losses_13044╦
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *+б(
&і#
input_1         ќ
є2Ѓ
3__inference_simple_conv_encoder_layer_call_fn_13098╦
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *+б(
&і#
input_1         ќ
В2ж
B__inference_flatten_layer_call_and_return_conditional_losses_13159б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Л2╬
'__inference_flatten_layer_call_fn_13164б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ж2у
@__inference_dense_layer_call_and_return_conditional_losses_13175б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
¤2╠
%__inference_dense_layer_call_fn_13184б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
╩BК
#__inference_signature_wrapper_13153input_1"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
в2У
A__inference_conv1d_layer_call_and_return_conditional_losses_13200б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
л2═
&__inference_conv1d_layer_call_fn_13209б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_conv1d_1_layer_call_and_return_conditional_losses_13225б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_conv1d_1_layer_call_fn_13234б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_conv1d_2_layer_call_and_return_conditional_losses_13250б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_conv1d_2_layer_call_fn_13259б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_conv1d_3_layer_call_and_return_conditional_losses_13275б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_conv1d_3_layer_call_fn_13284б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_conv1d_4_layer_call_and_return_conditional_losses_13300б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_conv1d_4_layer_call_fn_13309б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_conv1d_5_layer_call_and_return_conditional_losses_13325б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_conv1d_5_layer_call_fn_13334б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_conv1d_6_layer_call_and_return_conditional_losses_13350б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_conv1d_6_layer_call_fn_13359б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_conv1d_7_layer_call_and_return_conditional_losses_13375б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_conv1d_7_layer_call_fn_13384б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_conv1d_8_layer_call_and_return_conditional_losses_13400б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_conv1d_8_layer_call_fn_13409б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
ь2Ж
C__inference_conv1d_9_layer_call_and_return_conditional_losses_13425б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
м2¤
(__inference_conv1d_9_layer_call_fn_13434б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_conv1d_10_layer_call_and_return_conditional_losses_13450б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_conv1d_10_layer_call_fn_13459б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 Ф
 __inference__wrapped_model_12646є !"#$%&'()*+,-./01235б2
+б(
&і#
input_1         ќ
ф "3ф0
.
output_1"і
output_1         «
D__inference_conv1d_10_layer_call_and_return_conditional_losses_13450f234б1
*б'
%і"
inputs         ќ
ф "*б'
 і
0         ќ
џ є
)__inference_conv1d_10_layer_call_fn_13459Y234б1
*б'
%і"
inputs         ќ
ф "і         ќГ
C__inference_conv1d_1_layer_call_and_return_conditional_losses_13225f !4б1
*б'
%і"
inputs         ќ@
ф "*б'
 і
0         ќ@
џ Ё
(__inference_conv1d_1_layer_call_fn_13234Y !4б1
*б'
%і"
inputs         ќ@
ф "і         ќ@Г
C__inference_conv1d_2_layer_call_and_return_conditional_losses_13250f"#4б1
*б'
%і"
inputs         ќ@
ф "*б'
 і
0         ќ@
џ Ё
(__inference_conv1d_2_layer_call_fn_13259Y"#4б1
*б'
%і"
inputs         ќ@
ф "і         ќ@Г
C__inference_conv1d_3_layer_call_and_return_conditional_losses_13275f$%4б1
*б'
%і"
inputs         ќ@
ф "*б'
 і
0         ќ@
џ Ё
(__inference_conv1d_3_layer_call_fn_13284Y$%4б1
*б'
%і"
inputs         ќ@
ф "і         ќ@Г
C__inference_conv1d_4_layer_call_and_return_conditional_losses_13300f&'4б1
*б'
%і"
inputs         ќ@
ф "*б'
 і
0         ќ@
џ Ё
(__inference_conv1d_4_layer_call_fn_13309Y&'4б1
*б'
%і"
inputs         ќ@
ф "і         ќ@Г
C__inference_conv1d_5_layer_call_and_return_conditional_losses_13325f()4б1
*б'
%і"
inputs         ќ@
ф "*б'
 і
0         ќ@
џ Ё
(__inference_conv1d_5_layer_call_fn_13334Y()4б1
*б'
%і"
inputs         ќ@
ф "і         ќ@Г
C__inference_conv1d_6_layer_call_and_return_conditional_losses_13350f*+4б1
*б'
%і"
inputs         ќ@
ф "*б'
 і
0         ќ 
џ Ё
(__inference_conv1d_6_layer_call_fn_13359Y*+4б1
*б'
%і"
inputs         ќ@
ф "і         ќ Г
C__inference_conv1d_7_layer_call_and_return_conditional_losses_13375f,-4б1
*б'
%і"
inputs         ќ 
ф "*б'
 і
0         ќ
џ Ё
(__inference_conv1d_7_layer_call_fn_13384Y,-4б1
*б'
%і"
inputs         ќ 
ф "і         ќГ
C__inference_conv1d_8_layer_call_and_return_conditional_losses_13400f./4б1
*б'
%і"
inputs         ќ
ф "*б'
 і
0         ќ
џ Ё
(__inference_conv1d_8_layer_call_fn_13409Y./4б1
*б'
%і"
inputs         ќ
ф "і         ќГ
C__inference_conv1d_9_layer_call_and_return_conditional_losses_13425f014б1
*б'
%і"
inputs         ќ
ф "*б'
 і
0         ќ
џ Ё
(__inference_conv1d_9_layer_call_fn_13434Y014б1
*б'
%і"
inputs         ќ
ф "і         ќФ
A__inference_conv1d_layer_call_and_return_conditional_losses_13200f4б1
*б'
%і"
inputs         ќ
ф "*б'
 і
0         ќ@
џ Ѓ
&__inference_conv1d_layer_call_fn_13209Y4б1
*б'
%і"
inputs         ќ
ф "і         ќ@А
@__inference_dense_layer_call_and_return_conditional_losses_13175]0б-
&б#
!і
inputs         г
ф "%б"
і
0         
џ y
%__inference_dense_layer_call_fn_13184P0б-
&б#
!і
inputs         г
ф "і         ц
B__inference_flatten_layer_call_and_return_conditional_losses_13159^4б1
*б'
%і"
inputs         ќ
ф "&б#
і
0         г
џ |
'__inference_flatten_layer_call_fn_13164Q4б1
*б'
%і"
inputs         ќ
ф "і         г╣
#__inference_signature_wrapper_13153Љ !"#$%&'()*+,-./0123@б=
б 
6ф3
1
input_1&і#
input_1         ќ"3ф0
.
output_1"і
output_1         ╩
N__inference_simple_conv_encoder_layer_call_and_return_conditional_losses_13044x !"#$%&'()*+,-./01235б2
+б(
&і#
input_1         ќ
ф "%б"
і
0         
џ б
3__inference_simple_conv_encoder_layer_call_fn_13098k !"#$%&'()*+,-./01235б2
+б(
&і#
input_1         ќ
ф "і         