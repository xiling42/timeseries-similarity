��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
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
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
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
delete_old_dirsbool(�
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
�
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
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
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
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12v2.4.0-49-g85c8b2a817f8��

�
"simple_conv_decoder/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�K*3
shared_name$"simple_conv_decoder/dense_1/kernel
�
6simple_conv_decoder/dense_1/kernel/Read/ReadVariableOpReadVariableOp"simple_conv_decoder/dense_1/kernel*
_output_shapes
:	�K*
dtype0
�
 simple_conv_decoder/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�K*1
shared_name" simple_conv_decoder/dense_1/bias
�
4simple_conv_decoder/dense_1/bias/Read/ReadVariableOpReadVariableOp simple_conv_decoder/dense_1/bias*
_output_shapes	
:�K*
dtype0
�
"simple_conv_decoder/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*3
shared_name$"simple_conv_decoder/dense_2/kernel
�
6simple_conv_decoder/dense_2/kernel/Read/ReadVariableOpReadVariableOp"simple_conv_decoder/dense_2/kernel*
_output_shapes

:@*
dtype0
�
 simple_conv_decoder/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" simple_conv_decoder/dense_2/bias
�
4simple_conv_decoder/dense_2/bias/Read/ReadVariableOpReadVariableOp simple_conv_decoder/dense_2/bias*
_output_shapes
:*
dtype0
�
+simple_conv_decoder/conv1d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*<
shared_name-+simple_conv_decoder/conv1d_transpose/kernel
�
?simple_conv_decoder/conv1d_transpose/kernel/Read/ReadVariableOpReadVariableOp+simple_conv_decoder/conv1d_transpose/kernel*"
_output_shapes
:@@*
dtype0
�
)simple_conv_decoder/conv1d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)simple_conv_decoder/conv1d_transpose/bias
�
=simple_conv_decoder/conv1d_transpose/bias/Read/ReadVariableOpReadVariableOp)simple_conv_decoder/conv1d_transpose/bias*
_output_shapes
:@*
dtype0
�
-simple_conv_decoder/conv1d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*>
shared_name/-simple_conv_decoder/conv1d_transpose_1/kernel
�
Asimple_conv_decoder/conv1d_transpose_1/kernel/Read/ReadVariableOpReadVariableOp-simple_conv_decoder/conv1d_transpose_1/kernel*"
_output_shapes
:@@*
dtype0
�
+simple_conv_decoder/conv1d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+simple_conv_decoder/conv1d_transpose_1/bias
�
?simple_conv_decoder/conv1d_transpose_1/bias/Read/ReadVariableOpReadVariableOp+simple_conv_decoder/conv1d_transpose_1/bias*
_output_shapes
:@*
dtype0
�
-simple_conv_decoder/conv1d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*>
shared_name/-simple_conv_decoder/conv1d_transpose_2/kernel
�
Asimple_conv_decoder/conv1d_transpose_2/kernel/Read/ReadVariableOpReadVariableOp-simple_conv_decoder/conv1d_transpose_2/kernel*"
_output_shapes
:@@*
dtype0
�
+simple_conv_decoder/conv1d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+simple_conv_decoder/conv1d_transpose_2/bias
�
?simple_conv_decoder/conv1d_transpose_2/bias/Read/ReadVariableOpReadVariableOp+simple_conv_decoder/conv1d_transpose_2/bias*
_output_shapes
:@*
dtype0
�
-simple_conv_decoder/conv1d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*>
shared_name/-simple_conv_decoder/conv1d_transpose_3/kernel
�
Asimple_conv_decoder/conv1d_transpose_3/kernel/Read/ReadVariableOpReadVariableOp-simple_conv_decoder/conv1d_transpose_3/kernel*"
_output_shapes
:@@*
dtype0
�
+simple_conv_decoder/conv1d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+simple_conv_decoder/conv1d_transpose_3/bias
�
?simple_conv_decoder/conv1d_transpose_3/bias/Read/ReadVariableOpReadVariableOp+simple_conv_decoder/conv1d_transpose_3/bias*
_output_shapes
:@*
dtype0
�
-simple_conv_decoder/conv1d_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*>
shared_name/-simple_conv_decoder/conv1d_transpose_4/kernel
�
Asimple_conv_decoder/conv1d_transpose_4/kernel/Read/ReadVariableOpReadVariableOp-simple_conv_decoder/conv1d_transpose_4/kernel*"
_output_shapes
:@@*
dtype0
�
+simple_conv_decoder/conv1d_transpose_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+simple_conv_decoder/conv1d_transpose_4/bias
�
?simple_conv_decoder/conv1d_transpose_4/bias/Read/ReadVariableOpReadVariableOp+simple_conv_decoder/conv1d_transpose_4/bias*
_output_shapes
:@*
dtype0
�
-simple_conv_decoder/conv1d_transpose_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*>
shared_name/-simple_conv_decoder/conv1d_transpose_5/kernel
�
Asimple_conv_decoder/conv1d_transpose_5/kernel/Read/ReadVariableOpReadVariableOp-simple_conv_decoder/conv1d_transpose_5/kernel*"
_output_shapes
:@@*
dtype0
�
+simple_conv_decoder/conv1d_transpose_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*<
shared_name-+simple_conv_decoder/conv1d_transpose_5/bias
�
?simple_conv_decoder/conv1d_transpose_5/bias/Read/ReadVariableOpReadVariableOp+simple_conv_decoder/conv1d_transpose_5/bias*
_output_shapes
:@*
dtype0

NoOpNoOp
�*
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�)
value�)B�) B�)
�
deconvs

expand
reshape
out
regularization_losses
trainable_variables
	variables
	keras_api
	
signatures
*

0
1
2
3
4
5
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
 
v
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
12
13
14
15
v
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
12
13
14
15
�
,metrics
regularization_losses
-layer_regularization_losses
trainable_variables
.layer_metrics

/layers
0non_trainable_variables
	variables
 
h

 kernel
!bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
h

"kernel
#bias
5regularization_losses
6trainable_variables
7	variables
8	keras_api
h

$kernel
%bias
9regularization_losses
:trainable_variables
;	variables
<	keras_api
h

&kernel
'bias
=regularization_losses
>trainable_variables
?	variables
@	keras_api
h

(kernel
)bias
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
h

*kernel
+bias
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
`^
VARIABLE_VALUE"simple_conv_decoder/dense_1/kernel(expand/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUE simple_conv_decoder/dense_1/bias&expand/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
Imetrics
regularization_losses
Jlayer_regularization_losses
trainable_variables
Klayer_metrics

Llayers
Mnon_trainable_variables
	variables
 
 
 
�
Nmetrics
regularization_losses
Olayer_regularization_losses
trainable_variables
Player_metrics

Qlayers
Rnon_trainable_variables
	variables
][
VARIABLE_VALUE"simple_conv_decoder/dense_2/kernel%out/kernel/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUE simple_conv_decoder/dense_2/bias#out/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
Smetrics
regularization_losses
Tlayer_regularization_losses
trainable_variables
Ulayer_metrics

Vlayers
Wnon_trainable_variables
	variables
qo
VARIABLE_VALUE+simple_conv_decoder/conv1d_transpose/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUE)simple_conv_decoder/conv1d_transpose/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE-simple_conv_decoder/conv1d_transpose_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE+simple_conv_decoder/conv1d_transpose_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE-simple_conv_decoder/conv1d_transpose_2/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE+simple_conv_decoder/conv1d_transpose_2/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE-simple_conv_decoder/conv1d_transpose_3/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE+simple_conv_decoder/conv1d_transpose_3/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE-simple_conv_decoder/conv1d_transpose_4/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE+simple_conv_decoder/conv1d_transpose_4/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUE-simple_conv_decoder/conv1d_transpose_5/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE+simple_conv_decoder/conv1d_transpose_5/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
?

0
1
2
3
4
5
6
7
8
 
 

 0
!1

 0
!1
�
Xmetrics
1regularization_losses
Ylayer_regularization_losses
2trainable_variables
Zlayer_metrics

[layers
\non_trainable_variables
3	variables
 

"0
#1

"0
#1
�
]metrics
5regularization_losses
^layer_regularization_losses
6trainable_variables
_layer_metrics

`layers
anon_trainable_variables
7	variables
 

$0
%1

$0
%1
�
bmetrics
9regularization_losses
clayer_regularization_losses
:trainable_variables
dlayer_metrics

elayers
fnon_trainable_variables
;	variables
 

&0
'1

&0
'1
�
gmetrics
=regularization_losses
hlayer_regularization_losses
>trainable_variables
ilayer_metrics

jlayers
knon_trainable_variables
?	variables
 

(0
)1

(0
)1
�
lmetrics
Aregularization_losses
mlayer_regularization_losses
Btrainable_variables
nlayer_metrics

olayers
pnon_trainable_variables
C	variables
 

*0
+1

*0
+1
�
qmetrics
Eregularization_losses
rlayer_regularization_losses
Ftrainable_variables
slayer_metrics

tlayers
unon_trainable_variables
G	variables
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
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1"simple_conv_decoder/dense_1/kernel simple_conv_decoder/dense_1/bias+simple_conv_decoder/conv1d_transpose/kernel)simple_conv_decoder/conv1d_transpose/bias-simple_conv_decoder/conv1d_transpose_1/kernel+simple_conv_decoder/conv1d_transpose_1/bias-simple_conv_decoder/conv1d_transpose_2/kernel+simple_conv_decoder/conv1d_transpose_2/bias-simple_conv_decoder/conv1d_transpose_3/kernel+simple_conv_decoder/conv1d_transpose_3/bias-simple_conv_decoder/conv1d_transpose_4/kernel+simple_conv_decoder/conv1d_transpose_4/bias-simple_conv_decoder/conv1d_transpose_5/kernel+simple_conv_decoder/conv1d_transpose_5/bias"simple_conv_decoder/dense_2/kernel simple_conv_decoder/dense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_14520
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename6simple_conv_decoder/dense_1/kernel/Read/ReadVariableOp4simple_conv_decoder/dense_1/bias/Read/ReadVariableOp6simple_conv_decoder/dense_2/kernel/Read/ReadVariableOp4simple_conv_decoder/dense_2/bias/Read/ReadVariableOp?simple_conv_decoder/conv1d_transpose/kernel/Read/ReadVariableOp=simple_conv_decoder/conv1d_transpose/bias/Read/ReadVariableOpAsimple_conv_decoder/conv1d_transpose_1/kernel/Read/ReadVariableOp?simple_conv_decoder/conv1d_transpose_1/bias/Read/ReadVariableOpAsimple_conv_decoder/conv1d_transpose_2/kernel/Read/ReadVariableOp?simple_conv_decoder/conv1d_transpose_2/bias/Read/ReadVariableOpAsimple_conv_decoder/conv1d_transpose_3/kernel/Read/ReadVariableOp?simple_conv_decoder/conv1d_transpose_3/bias/Read/ReadVariableOpAsimple_conv_decoder/conv1d_transpose_4/kernel/Read/ReadVariableOp?simple_conv_decoder/conv1d_transpose_4/bias/Read/ReadVariableOpAsimple_conv_decoder/conv1d_transpose_5/kernel/Read/ReadVariableOp?simple_conv_decoder/conv1d_transpose_5/bias/Read/ReadVariableOpConst*
Tin
2*
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
GPU 2J 8� *'
f"R 
__inference__traced_save_14667
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename"simple_conv_decoder/dense_1/kernel simple_conv_decoder/dense_1/bias"simple_conv_decoder/dense_2/kernel simple_conv_decoder/dense_2/bias+simple_conv_decoder/conv1d_transpose/kernel)simple_conv_decoder/conv1d_transpose/bias-simple_conv_decoder/conv1d_transpose_1/kernel+simple_conv_decoder/conv1d_transpose_1/bias-simple_conv_decoder/conv1d_transpose_2/kernel+simple_conv_decoder/conv1d_transpose_2/bias-simple_conv_decoder/conv1d_transpose_3/kernel+simple_conv_decoder/conv1d_transpose_3/bias-simple_conv_decoder/conv1d_transpose_4/kernel+simple_conv_decoder/conv1d_transpose_4/bias-simple_conv_decoder/conv1d_transpose_5/kernel+simple_conv_decoder/conv1d_transpose_5/bias*
Tin
2*
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
GPU 2J 8� **
f%R#
!__inference__traced_restore_14725��	
�0
�

__inference__traced_save_14667
file_prefixA
=savev2_simple_conv_decoder_dense_1_kernel_read_readvariableop?
;savev2_simple_conv_decoder_dense_1_bias_read_readvariableopA
=savev2_simple_conv_decoder_dense_2_kernel_read_readvariableop?
;savev2_simple_conv_decoder_dense_2_bias_read_readvariableopJ
Fsavev2_simple_conv_decoder_conv1d_transpose_kernel_read_readvariableopH
Dsavev2_simple_conv_decoder_conv1d_transpose_bias_read_readvariableopL
Hsavev2_simple_conv_decoder_conv1d_transpose_1_kernel_read_readvariableopJ
Fsavev2_simple_conv_decoder_conv1d_transpose_1_bias_read_readvariableopL
Hsavev2_simple_conv_decoder_conv1d_transpose_2_kernel_read_readvariableopJ
Fsavev2_simple_conv_decoder_conv1d_transpose_2_bias_read_readvariableopL
Hsavev2_simple_conv_decoder_conv1d_transpose_3_kernel_read_readvariableopJ
Fsavev2_simple_conv_decoder_conv1d_transpose_3_bias_read_readvariableopL
Hsavev2_simple_conv_decoder_conv1d_transpose_4_kernel_read_readvariableopJ
Fsavev2_simple_conv_decoder_conv1d_transpose_4_bias_read_readvariableopL
Hsavev2_simple_conv_decoder_conv1d_transpose_5_kernel_read_readvariableopJ
Fsavev2_simple_conv_decoder_conv1d_transpose_5_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B(expand/kernel/.ATTRIBUTES/VARIABLE_VALUEB&expand/bias/.ATTRIBUTES/VARIABLE_VALUEB%out/kernel/.ATTRIBUTES/VARIABLE_VALUEB#out/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0=savev2_simple_conv_decoder_dense_1_kernel_read_readvariableop;savev2_simple_conv_decoder_dense_1_bias_read_readvariableop=savev2_simple_conv_decoder_dense_2_kernel_read_readvariableop;savev2_simple_conv_decoder_dense_2_bias_read_readvariableopFsavev2_simple_conv_decoder_conv1d_transpose_kernel_read_readvariableopDsavev2_simple_conv_decoder_conv1d_transpose_bias_read_readvariableopHsavev2_simple_conv_decoder_conv1d_transpose_1_kernel_read_readvariableopFsavev2_simple_conv_decoder_conv1d_transpose_1_bias_read_readvariableopHsavev2_simple_conv_decoder_conv1d_transpose_2_kernel_read_readvariableopFsavev2_simple_conv_decoder_conv1d_transpose_2_bias_read_readvariableopHsavev2_simple_conv_decoder_conv1d_transpose_3_kernel_read_readvariableopFsavev2_simple_conv_decoder_conv1d_transpose_3_bias_read_readvariableopHsavev2_simple_conv_decoder_conv1d_transpose_4_kernel_read_readvariableopFsavev2_simple_conv_decoder_conv1d_transpose_4_bias_read_readvariableopHsavev2_simple_conv_decoder_conv1d_transpose_5_kernel_read_readvariableopFsavev2_simple_conv_decoder_conv1d_transpose_5_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :	�K:�K:@::@@:@:@@:@:@@:@:@@:@:@@:@:@@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	�K:!

_output_shapes	
:�K:$ 

_output_shapes

:@: 

_output_shapes
::($
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
:@@: 

_output_shapes
:@:

_output_shapes
: 
�
�
2__inference_conv1d_transpose_2_layer_call_fn_14162

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_141522
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
�
0__inference_conv1d_transpose_layer_call_fn_14060

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_conv1d_transpose_layer_call_and_return_conditional_losses_140502
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
� 
�
B__inference_dense_2_layer_call_and_return_conditional_losses_14587

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :������������������@2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :������������������2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
�
2__inference_conv1d_transpose_1_layer_call_fn_14111

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_141012
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
�
2__inference_conv1d_transpose_5_layer_call_fn_14315

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_conv1d_transpose_5_layer_call_and_return_conditional_losses_143052
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�2
�
N__inference_simple_conv_decoder_layer_call_and_return_conditional_losses_14443
input_1
dense_1_14340
dense_1_14342
conv1d_transpose_14366
conv1d_transpose_14368
conv1d_transpose_1_14371
conv1d_transpose_1_14373
conv1d_transpose_2_14376
conv1d_transpose_2_14378
conv1d_transpose_3_14381
conv1d_transpose_3_14383
conv1d_transpose_4_14386
conv1d_transpose_4_14388
conv1d_transpose_5_14391
conv1d_transpose_5_14393
dense_2_14437
dense_2_14439
identity��(conv1d_transpose/StatefulPartitionedCall�*conv1d_transpose_1/StatefulPartitionedCall�*conv1d_transpose_2/StatefulPartitionedCall�*conv1d_transpose_3/StatefulPartitionedCall�*conv1d_transpose_4/StatefulPartitionedCall�*conv1d_transpose_5/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_1_14340dense_1_14342*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������K*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_143292!
dense_1/StatefulPartitionedCall�
reshape/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_143582
reshape/PartitionedCall�
(conv1d_transpose/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0conv1d_transpose_14366conv1d_transpose_14368*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_conv1d_transpose_layer_call_and_return_conditional_losses_140502*
(conv1d_transpose/StatefulPartitionedCall�
*conv1d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv1d_transpose/StatefulPartitionedCall:output:0conv1d_transpose_1_14371conv1d_transpose_1_14373*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_141012,
*conv1d_transpose_1/StatefulPartitionedCall�
*conv1d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_1/StatefulPartitionedCall:output:0conv1d_transpose_2_14376conv1d_transpose_2_14378*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_141522,
*conv1d_transpose_2/StatefulPartitionedCall�
*conv1d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_2/StatefulPartitionedCall:output:0conv1d_transpose_3_14381conv1d_transpose_3_14383*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_142032,
*conv1d_transpose_3/StatefulPartitionedCall�
*conv1d_transpose_4/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_3/StatefulPartitionedCall:output:0conv1d_transpose_4_14386conv1d_transpose_4_14388*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_142542,
*conv1d_transpose_4/StatefulPartitionedCall�
*conv1d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_4/StatefulPartitionedCall:output:0conv1d_transpose_5_14391conv1d_transpose_5_14393*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_conv1d_transpose_5_layer_call_and_return_conditional_losses_143052,
*conv1d_transpose_5/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall3conv1d_transpose_5/StatefulPartitionedCall:output:0dense_2_14437dense_2_14439*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_144262!
dense_2/StatefulPartitionedCall�
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0)^conv1d_transpose/StatefulPartitionedCall+^conv1d_transpose_1/StatefulPartitionedCall+^conv1d_transpose_2/StatefulPartitionedCall+^conv1d_transpose_3/StatefulPartitionedCall+^conv1d_transpose_4/StatefulPartitionedCall+^conv1d_transpose_5/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:���������::::::::::::::::2T
(conv1d_transpose/StatefulPartitionedCall(conv1d_transpose/StatefulPartitionedCall2X
*conv1d_transpose_1/StatefulPartitionedCall*conv1d_transpose_1/StatefulPartitionedCall2X
*conv1d_transpose_2/StatefulPartitionedCall*conv1d_transpose_2/StatefulPartitionedCall2X
*conv1d_transpose_3/StatefulPartitionedCall*conv1d_transpose_3/StatefulPartitionedCall2X
*conv1d_transpose_4/StatefulPartitionedCall*conv1d_transpose_4/StatefulPartitionedCall2X
*conv1d_transpose_5/StatefulPartitionedCall*conv1d_transpose_5/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
|
'__inference_dense_2_layer_call_fn_14596

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_144262
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�1
�
K__inference_conv1d_transpose_layer_call_and_return_conditional_losses_14050

inputs9
5conv1d_transpose_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack�
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim�
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"������������������@2
conv1d_transpose/ExpandDims�
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp�
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim�
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_transpose/ExpandDims_1�
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack�
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1�
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2�
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice�
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack�
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1�
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2�
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1�
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis�
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat�
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"������������������@*
paddingSAME*
strides
2
conv1d_transpose�
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :������������������@*
squeeze_dims
2
conv1d_transpose/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@2	
BiasAdde
TanhTanhBiasAdd:output:0*
T0*4
_output_shapes"
 :������������������@2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*
T0*4
_output_shapes"
 :������������������@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
^
B__inference_reshape_layer_call_and_return_conditional_losses_14358

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicee
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape/shape/2�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapet
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:����������@2	
Reshapei
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������K:P L
(
_output_shapes
:����������K
 
_user_specified_nameinputs
�	
�
B__inference_dense_1_layer_call_and_return_conditional_losses_14329

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�K*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������K2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�K*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������K2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������K2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
3__inference_simple_conv_decoder_layer_call_fn_14481
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

unknown_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_simple_conv_decoder_layer_call_and_return_conditional_losses_144432
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:���������::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�1
�
M__inference_conv1d_transpose_5_layer_call_and_return_conditional_losses_14305

inputs9
5conv1d_transpose_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack�
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim�
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"������������������@2
conv1d_transpose/ExpandDims�
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp�
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim�
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_transpose/ExpandDims_1�
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack�
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1�
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2�
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice�
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack�
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1�
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2�
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1�
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis�
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat�
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"������������������@*
paddingSAME*
strides
2
conv1d_transpose�
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :������������������@*
squeeze_dims
2
conv1d_transpose/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@2	
BiasAdde
TanhTanhBiasAdd:output:0*
T0*4
_output_shapes"
 :������������������@2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*
T0*4
_output_shapes"
 :������������������@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�1
�
M__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_14254

inputs9
5conv1d_transpose_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack�
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim�
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"������������������@2
conv1d_transpose/ExpandDims�
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp�
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim�
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_transpose/ExpandDims_1�
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack�
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1�
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2�
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice�
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack�
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1�
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2�
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1�
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis�
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat�
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"������������������@*
paddingSAME*
strides
2
conv1d_transpose�
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :������������������@*
squeeze_dims
2
conv1d_transpose/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@2	
BiasAdde
TanhTanhBiasAdd:output:0*
T0*4
_output_shapes"
 :������������������@2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*
T0*4
_output_shapes"
 :������������������@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�J
�
!__inference__traced_restore_14725
file_prefix7
3assignvariableop_simple_conv_decoder_dense_1_kernel7
3assignvariableop_1_simple_conv_decoder_dense_1_bias9
5assignvariableop_2_simple_conv_decoder_dense_2_kernel7
3assignvariableop_3_simple_conv_decoder_dense_2_biasB
>assignvariableop_4_simple_conv_decoder_conv1d_transpose_kernel@
<assignvariableop_5_simple_conv_decoder_conv1d_transpose_biasD
@assignvariableop_6_simple_conv_decoder_conv1d_transpose_1_kernelB
>assignvariableop_7_simple_conv_decoder_conv1d_transpose_1_biasD
@assignvariableop_8_simple_conv_decoder_conv1d_transpose_2_kernelB
>assignvariableop_9_simple_conv_decoder_conv1d_transpose_2_biasE
Aassignvariableop_10_simple_conv_decoder_conv1d_transpose_3_kernelC
?assignvariableop_11_simple_conv_decoder_conv1d_transpose_3_biasE
Aassignvariableop_12_simple_conv_decoder_conv1d_transpose_4_kernelC
?assignvariableop_13_simple_conv_decoder_conv1d_transpose_4_biasE
Aassignvariableop_14_simple_conv_decoder_conv1d_transpose_5_kernelC
?assignvariableop_15_simple_conv_decoder_conv1d_transpose_5_bias
identity_17��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B(expand/kernel/.ATTRIBUTES/VARIABLE_VALUEB&expand/bias/.ATTRIBUTES/VARIABLE_VALUEB%out/kernel/.ATTRIBUTES/VARIABLE_VALUEB#out/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*5
value,B*B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*X
_output_shapesF
D:::::::::::::::::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp3assignvariableop_simple_conv_decoder_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp3assignvariableop_1_simple_conv_decoder_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp5assignvariableop_2_simple_conv_decoder_dense_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp3assignvariableop_3_simple_conv_decoder_dense_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp>assignvariableop_4_simple_conv_decoder_conv1d_transpose_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp<assignvariableop_5_simple_conv_decoder_conv1d_transpose_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp@assignvariableop_6_simple_conv_decoder_conv1d_transpose_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp>assignvariableop_7_simple_conv_decoder_conv1d_transpose_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp@assignvariableop_8_simple_conv_decoder_conv1d_transpose_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp>assignvariableop_9_simple_conv_decoder_conv1d_transpose_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpAassignvariableop_10_simple_conv_decoder_conv1d_transpose_3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp?assignvariableop_11_simple_conv_decoder_conv1d_transpose_3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpAassignvariableop_12_simple_conv_decoder_conv1d_transpose_4_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp?assignvariableop_13_simple_conv_decoder_conv1d_transpose_4_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOpAassignvariableop_14_simple_conv_decoder_conv1d_transpose_5_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp?assignvariableop_15_simple_conv_decoder_conv1d_transpose_5_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_159
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_16Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_16�
Identity_17IdentityIdentity_16:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_17"#
identity_17Identity_17:output:0*U
_input_shapesD
B: ::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152(
AssignVariableOp_2AssignVariableOp_22(
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
�1
�
M__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_14152

inputs9
5conv1d_transpose_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack�
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim�
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"������������������@2
conv1d_transpose/ExpandDims�
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp�
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim�
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_transpose/ExpandDims_1�
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack�
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1�
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2�
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice�
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack�
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1�
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2�
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1�
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis�
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat�
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"������������������@*
paddingSAME*
strides
2
conv1d_transpose�
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :������������������@*
squeeze_dims
2
conv1d_transpose/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@2	
BiasAdde
TanhTanhBiasAdd:output:0*
T0*4
_output_shapes"
 :������������������@2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*
T0*4
_output_shapes"
 :������������������@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
� 
�
B__inference_dense_2_layer_call_and_return_conditional_losses_14426

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOp�
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis�
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis�
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const�
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1�
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis�
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat�
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :������������������@2
Tensordot/transpose�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2
Tensordot/Reshape�
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis�
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������2
	Tensordot�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
T0*4
_output_shapes"
 :������������������2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
|
'__inference_dense_1_layer_call_fn_14539

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������K*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_143292
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:����������K2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
2__inference_conv1d_transpose_3_layer_call_fn_14213

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_142032
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
�
2__inference_conv1d_transpose_4_layer_call_fn_14264

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_142542
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :������������������@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
��
�
 __inference__wrapped_model_14009
input_1>
:simple_conv_decoder_dense_1_matmul_readvariableop_resource?
;simple_conv_decoder_dense_1_biasadd_readvariableop_resource^
Zsimple_conv_decoder_conv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resourceH
Dsimple_conv_decoder_conv1d_transpose_biasadd_readvariableop_resource`
\simple_conv_decoder_conv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resourceJ
Fsimple_conv_decoder_conv1d_transpose_1_biasadd_readvariableop_resource`
\simple_conv_decoder_conv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resourceJ
Fsimple_conv_decoder_conv1d_transpose_2_biasadd_readvariableop_resource`
\simple_conv_decoder_conv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resourceJ
Fsimple_conv_decoder_conv1d_transpose_3_biasadd_readvariableop_resource`
\simple_conv_decoder_conv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resourceJ
Fsimple_conv_decoder_conv1d_transpose_4_biasadd_readvariableop_resource`
\simple_conv_decoder_conv1d_transpose_5_conv1d_transpose_expanddims_1_readvariableop_resourceJ
Fsimple_conv_decoder_conv1d_transpose_5_biasadd_readvariableop_resourceA
=simple_conv_decoder_dense_2_tensordot_readvariableop_resource?
;simple_conv_decoder_dense_2_biasadd_readvariableop_resource
identity��;simple_conv_decoder/conv1d_transpose/BiasAdd/ReadVariableOp�Qsimple_conv_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp�=simple_conv_decoder/conv1d_transpose_1/BiasAdd/ReadVariableOp�Ssimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp�=simple_conv_decoder/conv1d_transpose_2/BiasAdd/ReadVariableOp�Ssimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp�=simple_conv_decoder/conv1d_transpose_3/BiasAdd/ReadVariableOp�Ssimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp�=simple_conv_decoder/conv1d_transpose_4/BiasAdd/ReadVariableOp�Ssimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp�=simple_conv_decoder/conv1d_transpose_5/BiasAdd/ReadVariableOp�Ssimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp�2simple_conv_decoder/dense_1/BiasAdd/ReadVariableOp�1simple_conv_decoder/dense_1/MatMul/ReadVariableOp�2simple_conv_decoder/dense_2/BiasAdd/ReadVariableOp�4simple_conv_decoder/dense_2/Tensordot/ReadVariableOp�
1simple_conv_decoder/dense_1/MatMul/ReadVariableOpReadVariableOp:simple_conv_decoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�K*
dtype023
1simple_conv_decoder/dense_1/MatMul/ReadVariableOp�
"simple_conv_decoder/dense_1/MatMulMatMulinput_19simple_conv_decoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������K2$
"simple_conv_decoder/dense_1/MatMul�
2simple_conv_decoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp;simple_conv_decoder_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:�K*
dtype024
2simple_conv_decoder/dense_1/BiasAdd/ReadVariableOp�
#simple_conv_decoder/dense_1/BiasAddBiasAdd,simple_conv_decoder/dense_1/MatMul:product:0:simple_conv_decoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������K2%
#simple_conv_decoder/dense_1/BiasAdd�
!simple_conv_decoder/reshape/ShapeShape,simple_conv_decoder/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:2#
!simple_conv_decoder/reshape/Shape�
/simple_conv_decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/simple_conv_decoder/reshape/strided_slice/stack�
1simple_conv_decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1simple_conv_decoder/reshape/strided_slice/stack_1�
1simple_conv_decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1simple_conv_decoder/reshape/strided_slice/stack_2�
)simple_conv_decoder/reshape/strided_sliceStridedSlice*simple_conv_decoder/reshape/Shape:output:08simple_conv_decoder/reshape/strided_slice/stack:output:0:simple_conv_decoder/reshape/strided_slice/stack_1:output:0:simple_conv_decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)simple_conv_decoder/reshape/strided_slice�
+simple_conv_decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�2-
+simple_conv_decoder/reshape/Reshape/shape/1�
+simple_conv_decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2-
+simple_conv_decoder/reshape/Reshape/shape/2�
)simple_conv_decoder/reshape/Reshape/shapePack2simple_conv_decoder/reshape/strided_slice:output:04simple_conv_decoder/reshape/Reshape/shape/1:output:04simple_conv_decoder/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2+
)simple_conv_decoder/reshape/Reshape/shape�
#simple_conv_decoder/reshape/ReshapeReshape,simple_conv_decoder/dense_1/BiasAdd:output:02simple_conv_decoder/reshape/Reshape/shape:output:0*
T0*,
_output_shapes
:����������@2%
#simple_conv_decoder/reshape/Reshape�
*simple_conv_decoder/conv1d_transpose/ShapeShape,simple_conv_decoder/reshape/Reshape:output:0*
T0*
_output_shapes
:2,
*simple_conv_decoder/conv1d_transpose/Shape�
8simple_conv_decoder/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8simple_conv_decoder/conv1d_transpose/strided_slice/stack�
:simple_conv_decoder/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:simple_conv_decoder/conv1d_transpose/strided_slice/stack_1�
:simple_conv_decoder/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:simple_conv_decoder/conv1d_transpose/strided_slice/stack_2�
2simple_conv_decoder/conv1d_transpose/strided_sliceStridedSlice3simple_conv_decoder/conv1d_transpose/Shape:output:0Asimple_conv_decoder/conv1d_transpose/strided_slice/stack:output:0Csimple_conv_decoder/conv1d_transpose/strided_slice/stack_1:output:0Csimple_conv_decoder/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2simple_conv_decoder/conv1d_transpose/strided_slice�
:simple_conv_decoder/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2<
:simple_conv_decoder/conv1d_transpose/strided_slice_1/stack�
<simple_conv_decoder/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<simple_conv_decoder/conv1d_transpose/strided_slice_1/stack_1�
<simple_conv_decoder/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<simple_conv_decoder/conv1d_transpose/strided_slice_1/stack_2�
4simple_conv_decoder/conv1d_transpose/strided_slice_1StridedSlice3simple_conv_decoder/conv1d_transpose/Shape:output:0Csimple_conv_decoder/conv1d_transpose/strided_slice_1/stack:output:0Esimple_conv_decoder/conv1d_transpose/strided_slice_1/stack_1:output:0Esimple_conv_decoder/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4simple_conv_decoder/conv1d_transpose/strided_slice_1�
*simple_conv_decoder/conv1d_transpose/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2,
*simple_conv_decoder/conv1d_transpose/mul/y�
(simple_conv_decoder/conv1d_transpose/mulMul=simple_conv_decoder/conv1d_transpose/strided_slice_1:output:03simple_conv_decoder/conv1d_transpose/mul/y:output:0*
T0*
_output_shapes
: 2*
(simple_conv_decoder/conv1d_transpose/mul�
,simple_conv_decoder/conv1d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2.
,simple_conv_decoder/conv1d_transpose/stack/2�
*simple_conv_decoder/conv1d_transpose/stackPack;simple_conv_decoder/conv1d_transpose/strided_slice:output:0,simple_conv_decoder/conv1d_transpose/mul:z:05simple_conv_decoder/conv1d_transpose/stack/2:output:0*
N*
T0*
_output_shapes
:2,
*simple_conv_decoder/conv1d_transpose/stack�
Dsimple_conv_decoder/conv1d_transpose/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2F
Dsimple_conv_decoder/conv1d_transpose/conv1d_transpose/ExpandDims/dim�
@simple_conv_decoder/conv1d_transpose/conv1d_transpose/ExpandDims
ExpandDims,simple_conv_decoder/reshape/Reshape:output:0Msimple_conv_decoder/conv1d_transpose/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@2B
@simple_conv_decoder/conv1d_transpose/conv1d_transpose/ExpandDims�
Qsimple_conv_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOpZsimple_conv_decoder_conv1d_transpose_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02S
Qsimple_conv_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp�
Fsimple_conv_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2H
Fsimple_conv_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/dim�
Bsimple_conv_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1
ExpandDimsYsimple_conv_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Osimple_conv_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2D
Bsimple_conv_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1�
Isimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2K
Isimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_slice/stack�
Ksimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2M
Ksimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_slice/stack_1�
Ksimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2M
Ksimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_slice/stack_2�
Csimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_sliceStridedSlice3simple_conv_decoder/conv1d_transpose/stack:output:0Rsimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_slice/stack:output:0Tsimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_slice/stack_1:output:0Tsimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2E
Csimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_slice�
Ksimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2M
Ksimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack�
Msimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2O
Msimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1�
Msimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2O
Msimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2�
Esimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1StridedSlice3simple_conv_decoder/conv1d_transpose/stack:output:0Tsimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack:output:0Vsimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_1:output:0Vsimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2G
Esimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1�
Esimple_conv_decoder/conv1d_transpose/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Esimple_conv_decoder/conv1d_transpose/conv1d_transpose/concat/values_1�
Asimple_conv_decoder/conv1d_transpose/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2C
Asimple_conv_decoder/conv1d_transpose/conv1d_transpose/concat/axis�
<simple_conv_decoder/conv1d_transpose/conv1d_transpose/concatConcatV2Lsimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_slice:output:0Nsimple_conv_decoder/conv1d_transpose/conv1d_transpose/concat/values_1:output:0Nsimple_conv_decoder/conv1d_transpose/conv1d_transpose/strided_slice_1:output:0Jsimple_conv_decoder/conv1d_transpose/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2>
<simple_conv_decoder/conv1d_transpose/conv1d_transpose/concat�
5simple_conv_decoder/conv1d_transpose/conv1d_transposeConv2DBackpropInputEsimple_conv_decoder/conv1d_transpose/conv1d_transpose/concat:output:0Ksimple_conv_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1:output:0Isimple_conv_decoder/conv1d_transpose/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"������������������@*
paddingSAME*
strides
27
5simple_conv_decoder/conv1d_transpose/conv1d_transpose�
=simple_conv_decoder/conv1d_transpose/conv1d_transpose/SqueezeSqueeze>simple_conv_decoder/conv1d_transpose/conv1d_transpose:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims
2?
=simple_conv_decoder/conv1d_transpose/conv1d_transpose/Squeeze�
;simple_conv_decoder/conv1d_transpose/BiasAdd/ReadVariableOpReadVariableOpDsimple_conv_decoder_conv1d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02=
;simple_conv_decoder/conv1d_transpose/BiasAdd/ReadVariableOp�
,simple_conv_decoder/conv1d_transpose/BiasAddBiasAddFsimple_conv_decoder/conv1d_transpose/conv1d_transpose/Squeeze:output:0Csimple_conv_decoder/conv1d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@2.
,simple_conv_decoder/conv1d_transpose/BiasAdd�
)simple_conv_decoder/conv1d_transpose/TanhTanh5simple_conv_decoder/conv1d_transpose/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2+
)simple_conv_decoder/conv1d_transpose/Tanh�
,simple_conv_decoder/conv1d_transpose_1/ShapeShape-simple_conv_decoder/conv1d_transpose/Tanh:y:0*
T0*
_output_shapes
:2.
,simple_conv_decoder/conv1d_transpose_1/Shape�
:simple_conv_decoder/conv1d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:simple_conv_decoder/conv1d_transpose_1/strided_slice/stack�
<simple_conv_decoder/conv1d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<simple_conv_decoder/conv1d_transpose_1/strided_slice/stack_1�
<simple_conv_decoder/conv1d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<simple_conv_decoder/conv1d_transpose_1/strided_slice/stack_2�
4simple_conv_decoder/conv1d_transpose_1/strided_sliceStridedSlice5simple_conv_decoder/conv1d_transpose_1/Shape:output:0Csimple_conv_decoder/conv1d_transpose_1/strided_slice/stack:output:0Esimple_conv_decoder/conv1d_transpose_1/strided_slice/stack_1:output:0Esimple_conv_decoder/conv1d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4simple_conv_decoder/conv1d_transpose_1/strided_slice�
<simple_conv_decoder/conv1d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<simple_conv_decoder/conv1d_transpose_1/strided_slice_1/stack�
>simple_conv_decoder/conv1d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>simple_conv_decoder/conv1d_transpose_1/strided_slice_1/stack_1�
>simple_conv_decoder/conv1d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>simple_conv_decoder/conv1d_transpose_1/strided_slice_1/stack_2�
6simple_conv_decoder/conv1d_transpose_1/strided_slice_1StridedSlice5simple_conv_decoder/conv1d_transpose_1/Shape:output:0Esimple_conv_decoder/conv1d_transpose_1/strided_slice_1/stack:output:0Gsimple_conv_decoder/conv1d_transpose_1/strided_slice_1/stack_1:output:0Gsimple_conv_decoder/conv1d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6simple_conv_decoder/conv1d_transpose_1/strided_slice_1�
,simple_conv_decoder/conv1d_transpose_1/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,simple_conv_decoder/conv1d_transpose_1/mul/y�
*simple_conv_decoder/conv1d_transpose_1/mulMul?simple_conv_decoder/conv1d_transpose_1/strided_slice_1:output:05simple_conv_decoder/conv1d_transpose_1/mul/y:output:0*
T0*
_output_shapes
: 2,
*simple_conv_decoder/conv1d_transpose_1/mul�
.simple_conv_decoder/conv1d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@20
.simple_conv_decoder/conv1d_transpose_1/stack/2�
,simple_conv_decoder/conv1d_transpose_1/stackPack=simple_conv_decoder/conv1d_transpose_1/strided_slice:output:0.simple_conv_decoder/conv1d_transpose_1/mul:z:07simple_conv_decoder/conv1d_transpose_1/stack/2:output:0*
N*
T0*
_output_shapes
:2.
,simple_conv_decoder/conv1d_transpose_1/stack�
Fsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2H
Fsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims/dim�
Bsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims
ExpandDims-simple_conv_decoder/conv1d_transpose/Tanh:y:0Osimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@2D
Bsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims�
Ssimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp\simple_conv_decoder_conv1d_transpose_1_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02U
Ssimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp�
Hsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dim�
Dsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1
ExpandDims[simple_conv_decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Qsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2F
Dsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1�
Ksimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2M
Ksimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_slice/stack�
Msimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2O
Msimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1�
Msimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2O
Msimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2�
Esimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_sliceStridedSlice5simple_conv_decoder/conv1d_transpose_1/stack:output:0Tsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_slice/stack:output:0Vsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_1:output:0Vsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2G
Esimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_slice�
Msimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2O
Msimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack�
Osimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Q
Osimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1�
Osimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Osimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2�
Gsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1StridedSlice5simple_conv_decoder/conv1d_transpose_1/stack:output:0Vsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack:output:0Xsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_1:output:0Xsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2I
Gsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1�
Gsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/concat/values_1�
Csimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Csimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/concat/axis�
>simple_conv_decoder/conv1d_transpose_1/conv1d_transpose/concatConcatV2Nsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_slice:output:0Psimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/concat/values_1:output:0Psimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/strided_slice_1:output:0Lsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2@
>simple_conv_decoder/conv1d_transpose_1/conv1d_transpose/concat�
7simple_conv_decoder/conv1d_transpose_1/conv1d_transposeConv2DBackpropInputGsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/concat:output:0Msimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1:output:0Ksimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"������������������@*
paddingSAME*
strides
29
7simple_conv_decoder/conv1d_transpose_1/conv1d_transpose�
?simple_conv_decoder/conv1d_transpose_1/conv1d_transpose/SqueezeSqueeze@simple_conv_decoder/conv1d_transpose_1/conv1d_transpose:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims
2A
?simple_conv_decoder/conv1d_transpose_1/conv1d_transpose/Squeeze�
=simple_conv_decoder/conv1d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpFsimple_conv_decoder_conv1d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02?
=simple_conv_decoder/conv1d_transpose_1/BiasAdd/ReadVariableOp�
.simple_conv_decoder/conv1d_transpose_1/BiasAddBiasAddHsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/Squeeze:output:0Esimple_conv_decoder/conv1d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@20
.simple_conv_decoder/conv1d_transpose_1/BiasAdd�
+simple_conv_decoder/conv1d_transpose_1/TanhTanh7simple_conv_decoder/conv1d_transpose_1/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2-
+simple_conv_decoder/conv1d_transpose_1/Tanh�
,simple_conv_decoder/conv1d_transpose_2/ShapeShape/simple_conv_decoder/conv1d_transpose_1/Tanh:y:0*
T0*
_output_shapes
:2.
,simple_conv_decoder/conv1d_transpose_2/Shape�
:simple_conv_decoder/conv1d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:simple_conv_decoder/conv1d_transpose_2/strided_slice/stack�
<simple_conv_decoder/conv1d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<simple_conv_decoder/conv1d_transpose_2/strided_slice/stack_1�
<simple_conv_decoder/conv1d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<simple_conv_decoder/conv1d_transpose_2/strided_slice/stack_2�
4simple_conv_decoder/conv1d_transpose_2/strided_sliceStridedSlice5simple_conv_decoder/conv1d_transpose_2/Shape:output:0Csimple_conv_decoder/conv1d_transpose_2/strided_slice/stack:output:0Esimple_conv_decoder/conv1d_transpose_2/strided_slice/stack_1:output:0Esimple_conv_decoder/conv1d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4simple_conv_decoder/conv1d_transpose_2/strided_slice�
<simple_conv_decoder/conv1d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<simple_conv_decoder/conv1d_transpose_2/strided_slice_1/stack�
>simple_conv_decoder/conv1d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>simple_conv_decoder/conv1d_transpose_2/strided_slice_1/stack_1�
>simple_conv_decoder/conv1d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>simple_conv_decoder/conv1d_transpose_2/strided_slice_1/stack_2�
6simple_conv_decoder/conv1d_transpose_2/strided_slice_1StridedSlice5simple_conv_decoder/conv1d_transpose_2/Shape:output:0Esimple_conv_decoder/conv1d_transpose_2/strided_slice_1/stack:output:0Gsimple_conv_decoder/conv1d_transpose_2/strided_slice_1/stack_1:output:0Gsimple_conv_decoder/conv1d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6simple_conv_decoder/conv1d_transpose_2/strided_slice_1�
,simple_conv_decoder/conv1d_transpose_2/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,simple_conv_decoder/conv1d_transpose_2/mul/y�
*simple_conv_decoder/conv1d_transpose_2/mulMul?simple_conv_decoder/conv1d_transpose_2/strided_slice_1:output:05simple_conv_decoder/conv1d_transpose_2/mul/y:output:0*
T0*
_output_shapes
: 2,
*simple_conv_decoder/conv1d_transpose_2/mul�
.simple_conv_decoder/conv1d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@20
.simple_conv_decoder/conv1d_transpose_2/stack/2�
,simple_conv_decoder/conv1d_transpose_2/stackPack=simple_conv_decoder/conv1d_transpose_2/strided_slice:output:0.simple_conv_decoder/conv1d_transpose_2/mul:z:07simple_conv_decoder/conv1d_transpose_2/stack/2:output:0*
N*
T0*
_output_shapes
:2.
,simple_conv_decoder/conv1d_transpose_2/stack�
Fsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2H
Fsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims/dim�
Bsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims
ExpandDims/simple_conv_decoder/conv1d_transpose_1/Tanh:y:0Osimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@2D
Bsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims�
Ssimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp\simple_conv_decoder_conv1d_transpose_2_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02U
Ssimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp�
Hsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim�
Dsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1
ExpandDims[simple_conv_decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Qsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2F
Dsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1�
Ksimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2M
Ksimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_slice/stack�
Msimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2O
Msimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1�
Msimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2O
Msimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2�
Esimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_sliceStridedSlice5simple_conv_decoder/conv1d_transpose_2/stack:output:0Tsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_slice/stack:output:0Vsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_1:output:0Vsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2G
Esimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_slice�
Msimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2O
Msimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack�
Osimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Q
Osimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1�
Osimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Osimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2�
Gsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1StridedSlice5simple_conv_decoder/conv1d_transpose_2/stack:output:0Vsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack:output:0Xsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_1:output:0Xsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2I
Gsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1�
Gsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/concat/values_1�
Csimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Csimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/concat/axis�
>simple_conv_decoder/conv1d_transpose_2/conv1d_transpose/concatConcatV2Nsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_slice:output:0Psimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/concat/values_1:output:0Psimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/strided_slice_1:output:0Lsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2@
>simple_conv_decoder/conv1d_transpose_2/conv1d_transpose/concat�
7simple_conv_decoder/conv1d_transpose_2/conv1d_transposeConv2DBackpropInputGsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/concat:output:0Msimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1:output:0Ksimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"������������������@*
paddingSAME*
strides
29
7simple_conv_decoder/conv1d_transpose_2/conv1d_transpose�
?simple_conv_decoder/conv1d_transpose_2/conv1d_transpose/SqueezeSqueeze@simple_conv_decoder/conv1d_transpose_2/conv1d_transpose:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims
2A
?simple_conv_decoder/conv1d_transpose_2/conv1d_transpose/Squeeze�
=simple_conv_decoder/conv1d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpFsimple_conv_decoder_conv1d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02?
=simple_conv_decoder/conv1d_transpose_2/BiasAdd/ReadVariableOp�
.simple_conv_decoder/conv1d_transpose_2/BiasAddBiasAddHsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/Squeeze:output:0Esimple_conv_decoder/conv1d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@20
.simple_conv_decoder/conv1d_transpose_2/BiasAdd�
+simple_conv_decoder/conv1d_transpose_2/TanhTanh7simple_conv_decoder/conv1d_transpose_2/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2-
+simple_conv_decoder/conv1d_transpose_2/Tanh�
,simple_conv_decoder/conv1d_transpose_3/ShapeShape/simple_conv_decoder/conv1d_transpose_2/Tanh:y:0*
T0*
_output_shapes
:2.
,simple_conv_decoder/conv1d_transpose_3/Shape�
:simple_conv_decoder/conv1d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:simple_conv_decoder/conv1d_transpose_3/strided_slice/stack�
<simple_conv_decoder/conv1d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<simple_conv_decoder/conv1d_transpose_3/strided_slice/stack_1�
<simple_conv_decoder/conv1d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<simple_conv_decoder/conv1d_transpose_3/strided_slice/stack_2�
4simple_conv_decoder/conv1d_transpose_3/strided_sliceStridedSlice5simple_conv_decoder/conv1d_transpose_3/Shape:output:0Csimple_conv_decoder/conv1d_transpose_3/strided_slice/stack:output:0Esimple_conv_decoder/conv1d_transpose_3/strided_slice/stack_1:output:0Esimple_conv_decoder/conv1d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4simple_conv_decoder/conv1d_transpose_3/strided_slice�
<simple_conv_decoder/conv1d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<simple_conv_decoder/conv1d_transpose_3/strided_slice_1/stack�
>simple_conv_decoder/conv1d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>simple_conv_decoder/conv1d_transpose_3/strided_slice_1/stack_1�
>simple_conv_decoder/conv1d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>simple_conv_decoder/conv1d_transpose_3/strided_slice_1/stack_2�
6simple_conv_decoder/conv1d_transpose_3/strided_slice_1StridedSlice5simple_conv_decoder/conv1d_transpose_3/Shape:output:0Esimple_conv_decoder/conv1d_transpose_3/strided_slice_1/stack:output:0Gsimple_conv_decoder/conv1d_transpose_3/strided_slice_1/stack_1:output:0Gsimple_conv_decoder/conv1d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6simple_conv_decoder/conv1d_transpose_3/strided_slice_1�
,simple_conv_decoder/conv1d_transpose_3/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,simple_conv_decoder/conv1d_transpose_3/mul/y�
*simple_conv_decoder/conv1d_transpose_3/mulMul?simple_conv_decoder/conv1d_transpose_3/strided_slice_1:output:05simple_conv_decoder/conv1d_transpose_3/mul/y:output:0*
T0*
_output_shapes
: 2,
*simple_conv_decoder/conv1d_transpose_3/mul�
.simple_conv_decoder/conv1d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@20
.simple_conv_decoder/conv1d_transpose_3/stack/2�
,simple_conv_decoder/conv1d_transpose_3/stackPack=simple_conv_decoder/conv1d_transpose_3/strided_slice:output:0.simple_conv_decoder/conv1d_transpose_3/mul:z:07simple_conv_decoder/conv1d_transpose_3/stack/2:output:0*
N*
T0*
_output_shapes
:2.
,simple_conv_decoder/conv1d_transpose_3/stack�
Fsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2H
Fsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims/dim�
Bsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims
ExpandDims/simple_conv_decoder/conv1d_transpose_2/Tanh:y:0Osimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@2D
Bsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims�
Ssimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp\simple_conv_decoder_conv1d_transpose_3_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02U
Ssimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp�
Hsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim�
Dsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1
ExpandDims[simple_conv_decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Qsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2F
Dsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1�
Ksimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2M
Ksimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_slice/stack�
Msimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2O
Msimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1�
Msimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2O
Msimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2�
Esimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_sliceStridedSlice5simple_conv_decoder/conv1d_transpose_3/stack:output:0Tsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_slice/stack:output:0Vsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_1:output:0Vsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2G
Esimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_slice�
Msimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2O
Msimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack�
Osimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Q
Osimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1�
Osimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Osimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2�
Gsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1StridedSlice5simple_conv_decoder/conv1d_transpose_3/stack:output:0Vsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack:output:0Xsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_1:output:0Xsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2I
Gsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1�
Gsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/concat/values_1�
Csimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Csimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/concat/axis�
>simple_conv_decoder/conv1d_transpose_3/conv1d_transpose/concatConcatV2Nsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_slice:output:0Psimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/concat/values_1:output:0Psimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/strided_slice_1:output:0Lsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2@
>simple_conv_decoder/conv1d_transpose_3/conv1d_transpose/concat�
7simple_conv_decoder/conv1d_transpose_3/conv1d_transposeConv2DBackpropInputGsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/concat:output:0Msimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1:output:0Ksimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"������������������@*
paddingSAME*
strides
29
7simple_conv_decoder/conv1d_transpose_3/conv1d_transpose�
?simple_conv_decoder/conv1d_transpose_3/conv1d_transpose/SqueezeSqueeze@simple_conv_decoder/conv1d_transpose_3/conv1d_transpose:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims
2A
?simple_conv_decoder/conv1d_transpose_3/conv1d_transpose/Squeeze�
=simple_conv_decoder/conv1d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpFsimple_conv_decoder_conv1d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02?
=simple_conv_decoder/conv1d_transpose_3/BiasAdd/ReadVariableOp�
.simple_conv_decoder/conv1d_transpose_3/BiasAddBiasAddHsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/Squeeze:output:0Esimple_conv_decoder/conv1d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@20
.simple_conv_decoder/conv1d_transpose_3/BiasAdd�
+simple_conv_decoder/conv1d_transpose_3/TanhTanh7simple_conv_decoder/conv1d_transpose_3/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2-
+simple_conv_decoder/conv1d_transpose_3/Tanh�
,simple_conv_decoder/conv1d_transpose_4/ShapeShape/simple_conv_decoder/conv1d_transpose_3/Tanh:y:0*
T0*
_output_shapes
:2.
,simple_conv_decoder/conv1d_transpose_4/Shape�
:simple_conv_decoder/conv1d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:simple_conv_decoder/conv1d_transpose_4/strided_slice/stack�
<simple_conv_decoder/conv1d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<simple_conv_decoder/conv1d_transpose_4/strided_slice/stack_1�
<simple_conv_decoder/conv1d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<simple_conv_decoder/conv1d_transpose_4/strided_slice/stack_2�
4simple_conv_decoder/conv1d_transpose_4/strided_sliceStridedSlice5simple_conv_decoder/conv1d_transpose_4/Shape:output:0Csimple_conv_decoder/conv1d_transpose_4/strided_slice/stack:output:0Esimple_conv_decoder/conv1d_transpose_4/strided_slice/stack_1:output:0Esimple_conv_decoder/conv1d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4simple_conv_decoder/conv1d_transpose_4/strided_slice�
<simple_conv_decoder/conv1d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<simple_conv_decoder/conv1d_transpose_4/strided_slice_1/stack�
>simple_conv_decoder/conv1d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>simple_conv_decoder/conv1d_transpose_4/strided_slice_1/stack_1�
>simple_conv_decoder/conv1d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>simple_conv_decoder/conv1d_transpose_4/strided_slice_1/stack_2�
6simple_conv_decoder/conv1d_transpose_4/strided_slice_1StridedSlice5simple_conv_decoder/conv1d_transpose_4/Shape:output:0Esimple_conv_decoder/conv1d_transpose_4/strided_slice_1/stack:output:0Gsimple_conv_decoder/conv1d_transpose_4/strided_slice_1/stack_1:output:0Gsimple_conv_decoder/conv1d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6simple_conv_decoder/conv1d_transpose_4/strided_slice_1�
,simple_conv_decoder/conv1d_transpose_4/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,simple_conv_decoder/conv1d_transpose_4/mul/y�
*simple_conv_decoder/conv1d_transpose_4/mulMul?simple_conv_decoder/conv1d_transpose_4/strided_slice_1:output:05simple_conv_decoder/conv1d_transpose_4/mul/y:output:0*
T0*
_output_shapes
: 2,
*simple_conv_decoder/conv1d_transpose_4/mul�
.simple_conv_decoder/conv1d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@20
.simple_conv_decoder/conv1d_transpose_4/stack/2�
,simple_conv_decoder/conv1d_transpose_4/stackPack=simple_conv_decoder/conv1d_transpose_4/strided_slice:output:0.simple_conv_decoder/conv1d_transpose_4/mul:z:07simple_conv_decoder/conv1d_transpose_4/stack/2:output:0*
N*
T0*
_output_shapes
:2.
,simple_conv_decoder/conv1d_transpose_4/stack�
Fsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2H
Fsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/ExpandDims/dim�
Bsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/ExpandDims
ExpandDims/simple_conv_decoder/conv1d_transpose_3/Tanh:y:0Osimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@2D
Bsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/ExpandDims�
Ssimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp\simple_conv_decoder_conv1d_transpose_4_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02U
Ssimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp�
Hsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim�
Dsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/ExpandDims_1
ExpandDims[simple_conv_decoder/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Qsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2F
Dsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/ExpandDims_1�
Ksimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2M
Ksimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_slice/stack�
Msimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2O
Msimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1�
Msimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2O
Msimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2�
Esimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_sliceStridedSlice5simple_conv_decoder/conv1d_transpose_4/stack:output:0Tsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_slice/stack:output:0Vsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_1:output:0Vsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2G
Esimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_slice�
Msimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2O
Msimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack�
Osimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Q
Osimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1�
Osimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Osimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2�
Gsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_slice_1StridedSlice5simple_conv_decoder/conv1d_transpose_4/stack:output:0Vsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack:output:0Xsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_1:output:0Xsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2I
Gsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_slice_1�
Gsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/concat/values_1�
Csimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Csimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/concat/axis�
>simple_conv_decoder/conv1d_transpose_4/conv1d_transpose/concatConcatV2Nsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_slice:output:0Psimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/concat/values_1:output:0Psimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/strided_slice_1:output:0Lsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2@
>simple_conv_decoder/conv1d_transpose_4/conv1d_transpose/concat�
7simple_conv_decoder/conv1d_transpose_4/conv1d_transposeConv2DBackpropInputGsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/concat:output:0Msimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/ExpandDims_1:output:0Ksimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"������������������@*
paddingSAME*
strides
29
7simple_conv_decoder/conv1d_transpose_4/conv1d_transpose�
?simple_conv_decoder/conv1d_transpose_4/conv1d_transpose/SqueezeSqueeze@simple_conv_decoder/conv1d_transpose_4/conv1d_transpose:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims
2A
?simple_conv_decoder/conv1d_transpose_4/conv1d_transpose/Squeeze�
=simple_conv_decoder/conv1d_transpose_4/BiasAdd/ReadVariableOpReadVariableOpFsimple_conv_decoder_conv1d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02?
=simple_conv_decoder/conv1d_transpose_4/BiasAdd/ReadVariableOp�
.simple_conv_decoder/conv1d_transpose_4/BiasAddBiasAddHsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/Squeeze:output:0Esimple_conv_decoder/conv1d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@20
.simple_conv_decoder/conv1d_transpose_4/BiasAdd�
+simple_conv_decoder/conv1d_transpose_4/TanhTanh7simple_conv_decoder/conv1d_transpose_4/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2-
+simple_conv_decoder/conv1d_transpose_4/Tanh�
,simple_conv_decoder/conv1d_transpose_5/ShapeShape/simple_conv_decoder/conv1d_transpose_4/Tanh:y:0*
T0*
_output_shapes
:2.
,simple_conv_decoder/conv1d_transpose_5/Shape�
:simple_conv_decoder/conv1d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:simple_conv_decoder/conv1d_transpose_5/strided_slice/stack�
<simple_conv_decoder/conv1d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<simple_conv_decoder/conv1d_transpose_5/strided_slice/stack_1�
<simple_conv_decoder/conv1d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<simple_conv_decoder/conv1d_transpose_5/strided_slice/stack_2�
4simple_conv_decoder/conv1d_transpose_5/strided_sliceStridedSlice5simple_conv_decoder/conv1d_transpose_5/Shape:output:0Csimple_conv_decoder/conv1d_transpose_5/strided_slice/stack:output:0Esimple_conv_decoder/conv1d_transpose_5/strided_slice/stack_1:output:0Esimple_conv_decoder/conv1d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4simple_conv_decoder/conv1d_transpose_5/strided_slice�
<simple_conv_decoder/conv1d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<simple_conv_decoder/conv1d_transpose_5/strided_slice_1/stack�
>simple_conv_decoder/conv1d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>simple_conv_decoder/conv1d_transpose_5/strided_slice_1/stack_1�
>simple_conv_decoder/conv1d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>simple_conv_decoder/conv1d_transpose_5/strided_slice_1/stack_2�
6simple_conv_decoder/conv1d_transpose_5/strided_slice_1StridedSlice5simple_conv_decoder/conv1d_transpose_5/Shape:output:0Esimple_conv_decoder/conv1d_transpose_5/strided_slice_1/stack:output:0Gsimple_conv_decoder/conv1d_transpose_5/strided_slice_1/stack_1:output:0Gsimple_conv_decoder/conv1d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6simple_conv_decoder/conv1d_transpose_5/strided_slice_1�
,simple_conv_decoder/conv1d_transpose_5/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2.
,simple_conv_decoder/conv1d_transpose_5/mul/y�
*simple_conv_decoder/conv1d_transpose_5/mulMul?simple_conv_decoder/conv1d_transpose_5/strided_slice_1:output:05simple_conv_decoder/conv1d_transpose_5/mul/y:output:0*
T0*
_output_shapes
: 2,
*simple_conv_decoder/conv1d_transpose_5/mul�
.simple_conv_decoder/conv1d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@20
.simple_conv_decoder/conv1d_transpose_5/stack/2�
,simple_conv_decoder/conv1d_transpose_5/stackPack=simple_conv_decoder/conv1d_transpose_5/strided_slice:output:0.simple_conv_decoder/conv1d_transpose_5/mul:z:07simple_conv_decoder/conv1d_transpose_5/stack/2:output:0*
N*
T0*
_output_shapes
:2.
,simple_conv_decoder/conv1d_transpose_5/stack�
Fsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2H
Fsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/ExpandDims/dim�
Bsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/ExpandDims
ExpandDims/simple_conv_decoder/conv1d_transpose_4/Tanh:y:0Osimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/ExpandDims/dim:output:0*
T0*0
_output_shapes
:����������@2D
Bsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/ExpandDims�
Ssimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp\simple_conv_decoder_conv1d_transpose_5_conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02U
Ssimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp�
Hsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2J
Hsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/dim�
Dsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/ExpandDims_1
ExpandDims[simple_conv_decoder/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0Qsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2F
Dsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/ExpandDims_1�
Ksimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2M
Ksimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_slice/stack�
Msimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2O
Msimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_1�
Msimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2O
Msimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_2�
Esimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_sliceStridedSlice5simple_conv_decoder/conv1d_transpose_5/stack:output:0Tsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_slice/stack:output:0Vsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_1:output:0Vsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2G
Esimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_slice�
Msimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2O
Msimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack�
Osimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2Q
Osimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_1�
Osimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2Q
Osimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_2�
Gsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_slice_1StridedSlice5simple_conv_decoder/conv1d_transpose_5/stack:output:0Vsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack:output:0Xsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_1:output:0Xsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2I
Gsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_slice_1�
Gsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/concat/values_1�
Csimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2E
Csimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/concat/axis�
>simple_conv_decoder/conv1d_transpose_5/conv1d_transpose/concatConcatV2Nsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_slice:output:0Psimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/concat/values_1:output:0Psimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/strided_slice_1:output:0Lsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2@
>simple_conv_decoder/conv1d_transpose_5/conv1d_transpose/concat�
7simple_conv_decoder/conv1d_transpose_5/conv1d_transposeConv2DBackpropInputGsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/concat:output:0Msimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/ExpandDims_1:output:0Ksimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"������������������@*
paddingSAME*
strides
29
7simple_conv_decoder/conv1d_transpose_5/conv1d_transpose�
?simple_conv_decoder/conv1d_transpose_5/conv1d_transpose/SqueezeSqueeze@simple_conv_decoder/conv1d_transpose_5/conv1d_transpose:output:0*
T0*,
_output_shapes
:����������@*
squeeze_dims
2A
?simple_conv_decoder/conv1d_transpose_5/conv1d_transpose/Squeeze�
=simple_conv_decoder/conv1d_transpose_5/BiasAdd/ReadVariableOpReadVariableOpFsimple_conv_decoder_conv1d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02?
=simple_conv_decoder/conv1d_transpose_5/BiasAdd/ReadVariableOp�
.simple_conv_decoder/conv1d_transpose_5/BiasAddBiasAddHsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/Squeeze:output:0Esimple_conv_decoder/conv1d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������@20
.simple_conv_decoder/conv1d_transpose_5/BiasAdd�
+simple_conv_decoder/conv1d_transpose_5/TanhTanh7simple_conv_decoder/conv1d_transpose_5/BiasAdd:output:0*
T0*,
_output_shapes
:����������@2-
+simple_conv_decoder/conv1d_transpose_5/Tanh�
4simple_conv_decoder/dense_2/Tensordot/ReadVariableOpReadVariableOp=simple_conv_decoder_dense_2_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype026
4simple_conv_decoder/dense_2/Tensordot/ReadVariableOp�
*simple_conv_decoder/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2,
*simple_conv_decoder/dense_2/Tensordot/axes�
*simple_conv_decoder/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2,
*simple_conv_decoder/dense_2/Tensordot/free�
+simple_conv_decoder/dense_2/Tensordot/ShapeShape/simple_conv_decoder/conv1d_transpose_5/Tanh:y:0*
T0*
_output_shapes
:2-
+simple_conv_decoder/dense_2/Tensordot/Shape�
3simple_conv_decoder/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 25
3simple_conv_decoder/dense_2/Tensordot/GatherV2/axis�
.simple_conv_decoder/dense_2/Tensordot/GatherV2GatherV24simple_conv_decoder/dense_2/Tensordot/Shape:output:03simple_conv_decoder/dense_2/Tensordot/free:output:0<simple_conv_decoder/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:20
.simple_conv_decoder/dense_2/Tensordot/GatherV2�
5simple_conv_decoder/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 27
5simple_conv_decoder/dense_2/Tensordot/GatherV2_1/axis�
0simple_conv_decoder/dense_2/Tensordot/GatherV2_1GatherV24simple_conv_decoder/dense_2/Tensordot/Shape:output:03simple_conv_decoder/dense_2/Tensordot/axes:output:0>simple_conv_decoder/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:22
0simple_conv_decoder/dense_2/Tensordot/GatherV2_1�
+simple_conv_decoder/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2-
+simple_conv_decoder/dense_2/Tensordot/Const�
*simple_conv_decoder/dense_2/Tensordot/ProdProd7simple_conv_decoder/dense_2/Tensordot/GatherV2:output:04simple_conv_decoder/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2,
*simple_conv_decoder/dense_2/Tensordot/Prod�
-simple_conv_decoder/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2/
-simple_conv_decoder/dense_2/Tensordot/Const_1�
,simple_conv_decoder/dense_2/Tensordot/Prod_1Prod9simple_conv_decoder/dense_2/Tensordot/GatherV2_1:output:06simple_conv_decoder/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2.
,simple_conv_decoder/dense_2/Tensordot/Prod_1�
1simple_conv_decoder/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 23
1simple_conv_decoder/dense_2/Tensordot/concat/axis�
,simple_conv_decoder/dense_2/Tensordot/concatConcatV23simple_conv_decoder/dense_2/Tensordot/free:output:03simple_conv_decoder/dense_2/Tensordot/axes:output:0:simple_conv_decoder/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2.
,simple_conv_decoder/dense_2/Tensordot/concat�
+simple_conv_decoder/dense_2/Tensordot/stackPack3simple_conv_decoder/dense_2/Tensordot/Prod:output:05simple_conv_decoder/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2-
+simple_conv_decoder/dense_2/Tensordot/stack�
/simple_conv_decoder/dense_2/Tensordot/transpose	Transpose/simple_conv_decoder/conv1d_transpose_5/Tanh:y:05simple_conv_decoder/dense_2/Tensordot/concat:output:0*
T0*,
_output_shapes
:����������@21
/simple_conv_decoder/dense_2/Tensordot/transpose�
-simple_conv_decoder/dense_2/Tensordot/ReshapeReshape3simple_conv_decoder/dense_2/Tensordot/transpose:y:04simple_conv_decoder/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:������������������2/
-simple_conv_decoder/dense_2/Tensordot/Reshape�
,simple_conv_decoder/dense_2/Tensordot/MatMulMatMul6simple_conv_decoder/dense_2/Tensordot/Reshape:output:0<simple_conv_decoder/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2.
,simple_conv_decoder/dense_2/Tensordot/MatMul�
-simple_conv_decoder/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-simple_conv_decoder/dense_2/Tensordot/Const_2�
3simple_conv_decoder/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 25
3simple_conv_decoder/dense_2/Tensordot/concat_1/axis�
.simple_conv_decoder/dense_2/Tensordot/concat_1ConcatV27simple_conv_decoder/dense_2/Tensordot/GatherV2:output:06simple_conv_decoder/dense_2/Tensordot/Const_2:output:0<simple_conv_decoder/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:20
.simple_conv_decoder/dense_2/Tensordot/concat_1�
%simple_conv_decoder/dense_2/TensordotReshape6simple_conv_decoder/dense_2/Tensordot/MatMul:product:07simple_conv_decoder/dense_2/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:����������2'
%simple_conv_decoder/dense_2/Tensordot�
2simple_conv_decoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp;simple_conv_decoder_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype024
2simple_conv_decoder/dense_2/BiasAdd/ReadVariableOp�
#simple_conv_decoder/dense_2/BiasAddBiasAdd.simple_conv_decoder/dense_2/Tensordot:output:0:simple_conv_decoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:����������2%
#simple_conv_decoder/dense_2/BiasAdd�	
IdentityIdentity,simple_conv_decoder/dense_2/BiasAdd:output:0<^simple_conv_decoder/conv1d_transpose/BiasAdd/ReadVariableOpR^simple_conv_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp>^simple_conv_decoder/conv1d_transpose_1/BiasAdd/ReadVariableOpT^simple_conv_decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp>^simple_conv_decoder/conv1d_transpose_2/BiasAdd/ReadVariableOpT^simple_conv_decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp>^simple_conv_decoder/conv1d_transpose_3/BiasAdd/ReadVariableOpT^simple_conv_decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp>^simple_conv_decoder/conv1d_transpose_4/BiasAdd/ReadVariableOpT^simple_conv_decoder/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp>^simple_conv_decoder/conv1d_transpose_5/BiasAdd/ReadVariableOpT^simple_conv_decoder/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp3^simple_conv_decoder/dense_1/BiasAdd/ReadVariableOp2^simple_conv_decoder/dense_1/MatMul/ReadVariableOp3^simple_conv_decoder/dense_2/BiasAdd/ReadVariableOp5^simple_conv_decoder/dense_2/Tensordot/ReadVariableOp*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:���������::::::::::::::::2z
;simple_conv_decoder/conv1d_transpose/BiasAdd/ReadVariableOp;simple_conv_decoder/conv1d_transpose/BiasAdd/ReadVariableOp2�
Qsimple_conv_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOpQsimple_conv_decoder/conv1d_transpose/conv1d_transpose/ExpandDims_1/ReadVariableOp2~
=simple_conv_decoder/conv1d_transpose_1/BiasAdd/ReadVariableOp=simple_conv_decoder/conv1d_transpose_1/BiasAdd/ReadVariableOp2�
Ssimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOpSsimple_conv_decoder/conv1d_transpose_1/conv1d_transpose/ExpandDims_1/ReadVariableOp2~
=simple_conv_decoder/conv1d_transpose_2/BiasAdd/ReadVariableOp=simple_conv_decoder/conv1d_transpose_2/BiasAdd/ReadVariableOp2�
Ssimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOpSsimple_conv_decoder/conv1d_transpose_2/conv1d_transpose/ExpandDims_1/ReadVariableOp2~
=simple_conv_decoder/conv1d_transpose_3/BiasAdd/ReadVariableOp=simple_conv_decoder/conv1d_transpose_3/BiasAdd/ReadVariableOp2�
Ssimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOpSsimple_conv_decoder/conv1d_transpose_3/conv1d_transpose/ExpandDims_1/ReadVariableOp2~
=simple_conv_decoder/conv1d_transpose_4/BiasAdd/ReadVariableOp=simple_conv_decoder/conv1d_transpose_4/BiasAdd/ReadVariableOp2�
Ssimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOpSsimple_conv_decoder/conv1d_transpose_4/conv1d_transpose/ExpandDims_1/ReadVariableOp2~
=simple_conv_decoder/conv1d_transpose_5/BiasAdd/ReadVariableOp=simple_conv_decoder/conv1d_transpose_5/BiasAdd/ReadVariableOp2�
Ssimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOpSsimple_conv_decoder/conv1d_transpose_5/conv1d_transpose/ExpandDims_1/ReadVariableOp2h
2simple_conv_decoder/dense_1/BiasAdd/ReadVariableOp2simple_conv_decoder/dense_1/BiasAdd/ReadVariableOp2f
1simple_conv_decoder/dense_1/MatMul/ReadVariableOp1simple_conv_decoder/dense_1/MatMul/ReadVariableOp2h
2simple_conv_decoder/dense_2/BiasAdd/ReadVariableOp2simple_conv_decoder/dense_2/BiasAdd/ReadVariableOp2l
4simple_conv_decoder/dense_2/Tensordot/ReadVariableOp4simple_conv_decoder/dense_2/Tensordot/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�1
�
M__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_14101

inputs9
5conv1d_transpose_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack�
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim�
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"������������������@2
conv1d_transpose/ExpandDims�
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp�
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim�
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_transpose/ExpandDims_1�
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack�
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1�
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2�
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice�
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack�
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1�
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2�
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1�
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis�
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat�
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"������������������@*
paddingSAME*
strides
2
conv1d_transpose�
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :������������������@*
squeeze_dims
2
conv1d_transpose/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@2	
BiasAdde
TanhTanhBiasAdd:output:0*
T0*4
_output_shapes"
 :������������������@2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*
T0*4
_output_shapes"
 :������������������@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
^
B__inference_reshape_layer_call_and_return_conditional_losses_14552

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicee
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :�2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :@2
Reshape/shape/2�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapet
ReshapeReshapeinputsReshape/shape:output:0*
T0*,
_output_shapes
:����������@2	
Reshapei
IdentityIdentityReshape:output:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������K:P L
(
_output_shapes
:����������K
 
_user_specified_nameinputs
�1
�
M__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_14203

inputs9
5conv1d_transpose_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�,conv1d_transpose/ExpandDims_1/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2�
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/2w
stackPackstrided_slice:output:0mul:z:0stack/2:output:0*
N*
T0*
_output_shapes
:2
stack�
conv1d_transpose/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2!
conv1d_transpose/ExpandDims/dim�
conv1d_transpose/ExpandDims
ExpandDimsinputs(conv1d_transpose/ExpandDims/dim:output:0*
T0*8
_output_shapes&
$:"������������������@2
conv1d_transpose/ExpandDims�
,conv1d_transpose/ExpandDims_1/ReadVariableOpReadVariableOp5conv1d_transpose_expanddims_1_readvariableop_resource*"
_output_shapes
:@@*
dtype02.
,conv1d_transpose/ExpandDims_1/ReadVariableOp�
!conv1d_transpose/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!conv1d_transpose/ExpandDims_1/dim�
conv1d_transpose/ExpandDims_1
ExpandDims4conv1d_transpose/ExpandDims_1/ReadVariableOp:value:0*conv1d_transpose/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:@@2
conv1d_transpose/ExpandDims_1�
$conv1d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv1d_transpose/strided_slice/stack�
&conv1d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_1�
&conv1d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice/stack_2�
conv1d_transpose/strided_sliceStridedSlicestack:output:0-conv1d_transpose/strided_slice/stack:output:0/conv1d_transpose/strided_slice/stack_1:output:0/conv1d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask2 
conv1d_transpose/strided_slice�
&conv1d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2(
&conv1d_transpose/strided_slice_1/stack�
(conv1d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2*
(conv1d_transpose/strided_slice_1/stack_1�
(conv1d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv1d_transpose/strided_slice_1/stack_2�
 conv1d_transpose/strided_slice_1StridedSlicestack:output:0/conv1d_transpose/strided_slice_1/stack:output:01conv1d_transpose/strided_slice_1/stack_1:output:01conv1d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2"
 conv1d_transpose/strided_slice_1�
 conv1d_transpose/concat/values_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 conv1d_transpose/concat/values_1~
conv1d_transpose/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d_transpose/concat/axis�
conv1d_transpose/concatConcatV2'conv1d_transpose/strided_slice:output:0)conv1d_transpose/concat/values_1:output:0)conv1d_transpose/strided_slice_1:output:0%conv1d_transpose/concat/axis:output:0*
N*
T0*
_output_shapes
:2
conv1d_transpose/concat�
conv1d_transposeConv2DBackpropInput conv1d_transpose/concat:output:0&conv1d_transpose/ExpandDims_1:output:0$conv1d_transpose/ExpandDims:output:0*
T0*8
_output_shapes&
$:"������������������@*
paddingSAME*
strides
2
conv1d_transpose�
conv1d_transpose/SqueezeSqueezeconv1d_transpose:output:0*
T0*4
_output_shapes"
 :������������������@*
squeeze_dims
2
conv1d_transpose/Squeeze�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAdd!conv1d_transpose/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@2	
BiasAdde
TanhTanhBiasAdd:output:0*
T0*4
_output_shapes"
 :������������������@2
Tanh�
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp-^conv1d_transpose/ExpandDims_1/ReadVariableOp*
T0*4
_output_shapes"
 :������������������@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2\
,conv1d_transpose/ExpandDims_1/ReadVariableOp,conv1d_transpose/ExpandDims_1/ReadVariableOp:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
C
'__inference_reshape_layer_call_fn_14557

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_143582
PartitionedCallq
IdentityIdentityPartitionedCall:output:0*
T0*,
_output_shapes
:����������@2

Identity"
identityIdentity:output:0*'
_input_shapes
:����������K:P L
(
_output_shapes
:����������K
 
_user_specified_nameinputs
�

�
#__inference_signature_wrapper_14520
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

unknown_14
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:����������*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_140092
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:����������2

Identity"
identityIdentity:output:0*f
_input_shapesU
S:���������::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�	
�
B__inference_dense_1_layer_call_and_return_conditional_losses_14530

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�K*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������K2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�K*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������K2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:����������K2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������A
output_15
StatefulPartitionedCall:0����������tensorflow/serving/predict:��
�
deconvs

expand
reshape
out
regularization_losses
trainable_variables
	variables
	keras_api
	
signatures
v_default_save_signature
*w&call_and_return_all_conditional_losses
x__call__"�
_tf_keras_model�{"class_name": "SimpleConvDecoder", "name": "simple_conv_decoder", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "SimpleConvDecoder"}}
J

0
1
2
3
4
5"
trackable_list_wrapper
�

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*y&call_and_return_all_conditional_losses
z__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 9600, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [282, 16]}}
�
regularization_losses
trainable_variables
	variables
	keras_api
*{&call_and_return_all_conditional_losses
|__call__"�
_tf_keras_layer�{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [150, 64]}}}
�

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*}&call_and_return_all_conditional_losses
~__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [282, 150, 64]}}
 "
trackable_list_wrapper
�
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
12
13
14
15"
trackable_list_wrapper
�
 0
!1
"2
#3
$4
%5
&6
'7
(8
)9
*10
+11
12
13
14
15"
trackable_list_wrapper
�
,metrics
regularization_losses
-layer_regularization_losses
trainable_variables
.layer_metrics

/layers
0non_trainable_variables
	variables
x__call__
v_default_save_signature
*w&call_and_return_all_conditional_losses
&w"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
signature_map
�


 kernel
!bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv1DTranspose", "name": "conv1d_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [282, 150, 64]}}
�


"kernel
#bias
5regularization_losses
6trainable_variables
7	variables
8	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv1DTranspose", "name": "conv1d_transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_transpose_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [282, 150, 64]}}
�


$kernel
%bias
9regularization_losses
:trainable_variables
;	variables
<	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv1DTranspose", "name": "conv1d_transpose_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_transpose_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [282, 150, 64]}}
�


&kernel
'bias
=regularization_losses
>trainable_variables
?	variables
@	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv1DTranspose", "name": "conv1d_transpose_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_transpose_3", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [282, 150, 64]}}
�


(kernel
)bias
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv1DTranspose", "name": "conv1d_transpose_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_transpose_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [282, 150, 64]}}
�


*kernel
+bias
Eregularization_losses
Ftrainable_variables
G	variables
H	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Conv1DTranspose", "name": "conv1d_transpose_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_transpose_5", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [282, 150, 64]}}
5:3	�K2"simple_conv_decoder/dense_1/kernel
/:-�K2 simple_conv_decoder/dense_1/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Imetrics
regularization_losses
Jlayer_regularization_losses
trainable_variables
Klayer_metrics

Llayers
Mnon_trainable_variables
	variables
z__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Nmetrics
regularization_losses
Olayer_regularization_losses
trainable_variables
Player_metrics

Qlayers
Rnon_trainable_variables
	variables
|__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
4:2@2"simple_conv_decoder/dense_2/kernel
.:,2 simple_conv_decoder/dense_2/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
Smetrics
regularization_losses
Tlayer_regularization_losses
trainable_variables
Ulayer_metrics

Vlayers
Wnon_trainable_variables
	variables
~__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
A:?@@2+simple_conv_decoder/conv1d_transpose/kernel
7:5@2)simple_conv_decoder/conv1d_transpose/bias
C:A@@2-simple_conv_decoder/conv1d_transpose_1/kernel
9:7@2+simple_conv_decoder/conv1d_transpose_1/bias
C:A@@2-simple_conv_decoder/conv1d_transpose_2/kernel
9:7@2+simple_conv_decoder/conv1d_transpose_2/bias
C:A@@2-simple_conv_decoder/conv1d_transpose_3/kernel
9:7@2+simple_conv_decoder/conv1d_transpose_3/bias
C:A@@2-simple_conv_decoder/conv1d_transpose_4/kernel
9:7@2+simple_conv_decoder/conv1d_transpose_4/bias
C:A@@2-simple_conv_decoder/conv1d_transpose_5/kernel
9:7@2+simple_conv_decoder/conv1d_transpose_5/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
_

0
1
2
3
4
5
6
7
8"
trackable_list_wrapper
 "
trackable_list_wrapper
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
�
Xmetrics
1regularization_losses
Ylayer_regularization_losses
2trainable_variables
Zlayer_metrics

[layers
\non_trainable_variables
3	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
]metrics
5regularization_losses
^layer_regularization_losses
6trainable_variables
_layer_metrics

`layers
anon_trainable_variables
7	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
bmetrics
9regularization_losses
clayer_regularization_losses
:trainable_variables
dlayer_metrics

elayers
fnon_trainable_variables
;	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
gmetrics
=regularization_losses
hlayer_regularization_losses
>trainable_variables
ilayer_metrics

jlayers
knon_trainable_variables
?	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
lmetrics
Aregularization_losses
mlayer_regularization_losses
Btrainable_variables
nlayer_metrics

olayers
pnon_trainable_variables
C	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�
qmetrics
Eregularization_losses
rlayer_regularization_losses
Ftrainable_variables
slayer_metrics

tlayers
unon_trainable_variables
G	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
�2�
 __inference__wrapped_model_14009�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�
input_1���������
�2�
N__inference_simple_conv_decoder_layer_call_and_return_conditional_losses_14443�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�
input_1���������
�2�
3__inference_simple_conv_decoder_layer_call_fn_14481�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�
input_1���������
�2�
B__inference_dense_1_layer_call_and_return_conditional_losses_14530�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_1_layer_call_fn_14539�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_reshape_layer_call_and_return_conditional_losses_14552�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_reshape_layer_call_fn_14557�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_2_layer_call_and_return_conditional_losses_14587�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_2_layer_call_fn_14596�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
#__inference_signature_wrapper_14520input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
K__inference_conv1d_transpose_layer_call_and_return_conditional_losses_14050�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"������������������@
�2�
0__inference_conv1d_transpose_layer_call_fn_14060�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"������������������@
�2�
M__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_14101�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"������������������@
�2�
2__inference_conv1d_transpose_1_layer_call_fn_14111�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"������������������@
�2�
M__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_14152�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"������������������@
�2�
2__inference_conv1d_transpose_2_layer_call_fn_14162�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"������������������@
�2�
M__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_14203�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"������������������@
�2�
2__inference_conv1d_transpose_3_layer_call_fn_14213�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"������������������@
�2�
M__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_14254�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"������������������@
�2�
2__inference_conv1d_transpose_4_layer_call_fn_14264�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"������������������@
�2�
M__inference_conv1d_transpose_5_layer_call_and_return_conditional_losses_14305�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"������������������@
�2�
2__inference_conv1d_transpose_5_layer_call_fn_14315�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� **�'
%�"������������������@�
 __inference__wrapped_model_14009~ !"#$%&'()*+0�-
&�#
!�
input_1���������
� "8�5
3
output_1'�$
output_1�����������
M__inference_conv1d_transpose_1_layer_call_and_return_conditional_losses_14101v"#<�9
2�/
-�*
inputs������������������@
� "2�/
(�%
0������������������@
� �
2__inference_conv1d_transpose_1_layer_call_fn_14111i"#<�9
2�/
-�*
inputs������������������@
� "%�"������������������@�
M__inference_conv1d_transpose_2_layer_call_and_return_conditional_losses_14152v$%<�9
2�/
-�*
inputs������������������@
� "2�/
(�%
0������������������@
� �
2__inference_conv1d_transpose_2_layer_call_fn_14162i$%<�9
2�/
-�*
inputs������������������@
� "%�"������������������@�
M__inference_conv1d_transpose_3_layer_call_and_return_conditional_losses_14203v&'<�9
2�/
-�*
inputs������������������@
� "2�/
(�%
0������������������@
� �
2__inference_conv1d_transpose_3_layer_call_fn_14213i&'<�9
2�/
-�*
inputs������������������@
� "%�"������������������@�
M__inference_conv1d_transpose_4_layer_call_and_return_conditional_losses_14254v()<�9
2�/
-�*
inputs������������������@
� "2�/
(�%
0������������������@
� �
2__inference_conv1d_transpose_4_layer_call_fn_14264i()<�9
2�/
-�*
inputs������������������@
� "%�"������������������@�
M__inference_conv1d_transpose_5_layer_call_and_return_conditional_losses_14305v*+<�9
2�/
-�*
inputs������������������@
� "2�/
(�%
0������������������@
� �
2__inference_conv1d_transpose_5_layer_call_fn_14315i*+<�9
2�/
-�*
inputs������������������@
� "%�"������������������@�
K__inference_conv1d_transpose_layer_call_and_return_conditional_losses_14050v !<�9
2�/
-�*
inputs������������������@
� "2�/
(�%
0������������������@
� �
0__inference_conv1d_transpose_layer_call_fn_14060i !<�9
2�/
-�*
inputs������������������@
� "%�"������������������@�
B__inference_dense_1_layer_call_and_return_conditional_losses_14530]/�,
%�"
 �
inputs���������
� "&�#
�
0����������K
� {
'__inference_dense_1_layer_call_fn_14539P/�,
%�"
 �
inputs���������
� "�����������K�
B__inference_dense_2_layer_call_and_return_conditional_losses_14587v<�9
2�/
-�*
inputs������������������@
� "2�/
(�%
0������������������
� �
'__inference_dense_2_layer_call_fn_14596i<�9
2�/
-�*
inputs������������������@
� "%�"�������������������
B__inference_reshape_layer_call_and_return_conditional_losses_14552^0�-
&�#
!�
inputs����������K
� "*�'
 �
0����������@
� |
'__inference_reshape_layer_call_fn_14557Q0�-
&�#
!�
inputs����������K
� "�����������@�
#__inference_signature_wrapper_14520� !"#$%&'()*+;�8
� 
1�.
,
input_1!�
input_1���������"8�5
3
output_1'�$
output_1�����������
N__inference_simple_conv_decoder_layer_call_and_return_conditional_losses_14443x !"#$%&'()*+0�-
&�#
!�
input_1���������
� "2�/
(�%
0������������������
� �
3__inference_simple_conv_decoder_layer_call_fn_14481k !"#$%&'()*+0�-
&�#
!�
input_1���������
� "%�"������������������