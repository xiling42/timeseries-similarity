бН
╤г
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
╛
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
executor_typestring И
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.3.02v2.3.0-rc2-23-gb36436b0878╡Ц
Ж
encoder/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
А·*%
shared_nameencoder/dense/kernel

(encoder/dense/kernel/Read/ReadVariableOpReadVariableOpencoder/dense/kernel* 
_output_shapes
:
А·*
dtype0
|
encoder/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameencoder/dense/bias
u
&encoder/dense/bias/Read/ReadVariableOpReadVariableOpencoder/dense/bias*
_output_shapes
:*
dtype0
К
encoder/conv1d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameencoder/conv1d/kernel
Г
)encoder/conv1d/kernel/Read/ReadVariableOpReadVariableOpencoder/conv1d/kernel*"
_output_shapes
:*
dtype0
~
encoder/conv1d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameencoder/conv1d/bias
w
'encoder/conv1d/bias/Read/ReadVariableOpReadVariableOpencoder/conv1d/bias*
_output_shapes
:*
dtype0
О
encoder/conv1d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameencoder/conv1d_1/kernel
З
+encoder/conv1d_1/kernel/Read/ReadVariableOpReadVariableOpencoder/conv1d_1/kernel*"
_output_shapes
: *
dtype0
В
encoder/conv1d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameencoder/conv1d_1/bias
{
)encoder/conv1d_1/bias/Read/ReadVariableOpReadVariableOpencoder/conv1d_1/bias*
_output_shapes
: *
dtype0
О
encoder/conv1d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*(
shared_nameencoder/conv1d_2/kernel
З
+encoder/conv1d_2/kernel/Read/ReadVariableOpReadVariableOpencoder/conv1d_2/kernel*"
_output_shapes
: @*
dtype0
В
encoder/conv1d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameencoder/conv1d_2/bias
{
)encoder/conv1d_2/bias/Read/ReadVariableOpReadVariableOpencoder/conv1d_2/bias*
_output_shapes
:@*
dtype0
Ъ
!encoder/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!encoder/batch_normalization/gamma
У
5encoder/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp!encoder/batch_normalization/gamma*
_output_shapes
:*
dtype0
Ш
 encoder/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" encoder/batch_normalization/beta
С
4encoder/batch_normalization/beta/Read/ReadVariableOpReadVariableOp encoder/batch_normalization/beta*
_output_shapes
:*
dtype0
Ю
#encoder/batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#encoder/batch_normalization_1/gamma
Ч
7encoder/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOp#encoder/batch_normalization_1/gamma*
_output_shapes
: *
dtype0
Ь
"encoder/batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"encoder/batch_normalization_1/beta
Х
6encoder/batch_normalization_1/beta/Read/ReadVariableOpReadVariableOp"encoder/batch_normalization_1/beta*
_output_shapes
: *
dtype0
Ю
#encoder/batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#encoder/batch_normalization_2/gamma
Ч
7encoder/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOp#encoder/batch_normalization_2/gamma*
_output_shapes
:@*
dtype0
Ь
"encoder/batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"encoder/batch_normalization_2/beta
Х
6encoder/batch_normalization_2/beta/Read/ReadVariableOpReadVariableOp"encoder/batch_normalization_2/beta*
_output_shapes
:@*
dtype0
ж
'encoder/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'encoder/batch_normalization/moving_mean
Я
;encoder/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp'encoder/batch_normalization/moving_mean*
_output_shapes
:*
dtype0
о
+encoder/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*<
shared_name-+encoder/batch_normalization/moving_variance
з
?encoder/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp+encoder/batch_normalization/moving_variance*
_output_shapes
:*
dtype0
к
)encoder/batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *:
shared_name+)encoder/batch_normalization_1/moving_mean
г
=encoder/batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp)encoder/batch_normalization_1/moving_mean*
_output_shapes
: *
dtype0
▓
-encoder/batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-encoder/batch_normalization_1/moving_variance
л
Aencoder/batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp-encoder/batch_normalization_1/moving_variance*
_output_shapes
: *
dtype0
к
)encoder/batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*:
shared_name+)encoder/batch_normalization_2/moving_mean
г
=encoder/batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp)encoder/batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
▓
-encoder/batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-encoder/batch_normalization_2/moving_variance
л
Aencoder/batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp-encoder/batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0

NoOpNoOp
ф+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Я+
valueХ+BТ+ BЛ+
О
	convs
	norms
flatten
out
regularization_losses
trainable_variables
	variables
	keras_api
	
signatures


0
1
2

0
1
2
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
 
f
0
1
2
3
4
5
 6
!7
"8
#9
$10
%11
12
13
Ц
0
1
2
3
4
5
 6
!7
"8
#9
$10
%11
&12
'13
(14
)15
*16
+17
18
19
н
,metrics
regularization_losses
-layer_metrics
.layer_regularization_losses
trainable_variables
/non_trainable_variables

0layers
	variables
 
h

kernel
bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
h

kernel
bias
5regularization_losses
6trainable_variables
7	variables
8	keras_api
h

kernel
bias
9regularization_losses
:trainable_variables
;	variables
<	keras_api
Ч
=axis
	 gamma
!beta
&moving_mean
'moving_variance
>regularization_losses
?trainable_variables
@	variables
A	keras_api
Ч
Baxis
	"gamma
#beta
(moving_mean
)moving_variance
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
Ч
Gaxis
	$gamma
%beta
*moving_mean
+moving_variance
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
 
 
 
н
Lmetrics
Mlayer_metrics
regularization_losses
Nlayer_regularization_losses
trainable_variables
Onon_trainable_variables

Players
	variables
OM
VARIABLE_VALUEencoder/dense/kernel%out/kernel/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEencoder/dense/bias#out/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
н
Qmetrics
Rlayer_metrics
regularization_losses
Slayer_regularization_losses
trainable_variables
Tnon_trainable_variables

Ulayers
	variables
[Y
VARIABLE_VALUEencoder/conv1d/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUEencoder/conv1d/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEencoder/conv1d_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEencoder/conv1d_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUEencoder/conv1d_2/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEencoder/conv1d_2/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUE!encoder/batch_normalization/gamma0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE encoder/batch_normalization/beta0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#encoder/batch_normalization_1/gamma0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE"encoder/batch_normalization_1/beta0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE#encoder/batch_normalization_2/gamma1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE"encoder/batch_normalization_2/beta1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUE'encoder/batch_normalization/moving_mean'variables/12/.ATTRIBUTES/VARIABLE_VALUE
hf
VARIABLE_VALUE+encoder/batch_normalization/moving_variance'variables/13/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)encoder/batch_normalization_1/moving_mean'variables/14/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-encoder/batch_normalization_1/moving_variance'variables/15/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUE)encoder/batch_normalization_2/moving_mean'variables/16/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE-encoder/batch_normalization_2/moving_variance'variables/17/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
*
&0
'1
(2
)3
*4
+5
8

0
1
2
3
4
5
6
7
 

0
1

0
1
н
Vmetrics
Wlayer_metrics
1regularization_losses
Xlayer_regularization_losses
2trainable_variables
Ynon_trainable_variables

Zlayers
3	variables
 

0
1

0
1
н
[metrics
\layer_metrics
5regularization_losses
]layer_regularization_losses
6trainable_variables
^non_trainable_variables

_layers
7	variables
 

0
1

0
1
н
`metrics
alayer_metrics
9regularization_losses
blayer_regularization_losses
:trainable_variables
cnon_trainable_variables

dlayers
;	variables
 
 

 0
!1

 0
!1
&2
'3
н
emetrics
flayer_metrics
>regularization_losses
glayer_regularization_losses
?trainable_variables
hnon_trainable_variables

ilayers
@	variables
 
 

"0
#1

"0
#1
(2
)3
н
jmetrics
klayer_metrics
Cregularization_losses
llayer_regularization_losses
Dtrainable_variables
mnon_trainable_variables

nlayers
E	variables
 
 

$0
%1

$0
%1
*2
+3
н
ometrics
player_metrics
Hregularization_losses
qlayer_regularization_losses
Itrainable_variables
rnon_trainable_variables

slayers
J	variables
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

&0
'1
 
 
 
 

(0
)1
 
 
 
 

*0
+1
 
Д
serving_default_input_1Placeholder*,
_output_shapes
:         А*
dtype0*!
shape:         А
Є
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1encoder/conv1d/kernelencoder/conv1d/bias+encoder/batch_normalization/moving_variance!encoder/batch_normalization/gamma'encoder/batch_normalization/moving_mean encoder/batch_normalization/betaencoder/conv1d_1/kernelencoder/conv1d_1/bias-encoder/batch_normalization_1/moving_variance#encoder/batch_normalization_1/gamma)encoder/batch_normalization_1/moving_mean"encoder/batch_normalization_1/betaencoder/conv1d_2/kernelencoder/conv1d_2/bias-encoder/batch_normalization_2/moving_variance#encoder/batch_normalization_2/gamma)encoder/batch_normalization_2/moving_mean"encoder/batch_normalization_2/betaencoder/dense/kernelencoder/dense/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *.
f)R'
%__inference_signature_wrapper_9200712
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
╫

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(encoder/dense/kernel/Read/ReadVariableOp&encoder/dense/bias/Read/ReadVariableOp)encoder/conv1d/kernel/Read/ReadVariableOp'encoder/conv1d/bias/Read/ReadVariableOp+encoder/conv1d_1/kernel/Read/ReadVariableOp)encoder/conv1d_1/bias/Read/ReadVariableOp+encoder/conv1d_2/kernel/Read/ReadVariableOp)encoder/conv1d_2/bias/Read/ReadVariableOp5encoder/batch_normalization/gamma/Read/ReadVariableOp4encoder/batch_normalization/beta/Read/ReadVariableOp7encoder/batch_normalization_1/gamma/Read/ReadVariableOp6encoder/batch_normalization_1/beta/Read/ReadVariableOp7encoder/batch_normalization_2/gamma/Read/ReadVariableOp6encoder/batch_normalization_2/beta/Read/ReadVariableOp;encoder/batch_normalization/moving_mean/Read/ReadVariableOp?encoder/batch_normalization/moving_variance/Read/ReadVariableOp=encoder/batch_normalization_1/moving_mean/Read/ReadVariableOpAencoder/batch_normalization_1/moving_variance/Read/ReadVariableOp=encoder/batch_normalization_2/moving_mean/Read/ReadVariableOpAencoder/batch_normalization_2/moving_variance/Read/ReadVariableOpConst*!
Tin
2*
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
GPU 2J 8В *)
f$R"
 __inference__traced_save_9202052
┬
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameencoder/dense/kernelencoder/dense/biasencoder/conv1d/kernelencoder/conv1d/biasencoder/conv1d_1/kernelencoder/conv1d_1/biasencoder/conv1d_2/kernelencoder/conv1d_2/bias!encoder/batch_normalization/gamma encoder/batch_normalization/beta#encoder/batch_normalization_1/gamma"encoder/batch_normalization_1/beta#encoder/batch_normalization_2/gamma"encoder/batch_normalization_2/beta'encoder/batch_normalization/moving_mean+encoder/batch_normalization/moving_variance)encoder/batch_normalization_1/moving_mean-encoder/batch_normalization_1/moving_variance)encoder/batch_normalization_2/moving_mean-encoder/batch_normalization_2/moving_variance* 
Tin
2*
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
GPU 2J 8В *,
f'R%
#__inference__traced_restore_9202122мЯ
╔
и
5__inference_batch_normalization_layer_call_fn_9201641

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         №*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_92000952
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         №2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         №::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         №
 
_user_specified_nameinputs
╞
Х
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9199853

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                   2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                   2
batchnorm/add_1t
IdentityIdentitybatchnorm/add_1:z:0*
T0*4
_output_shapes"
 :                   2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                   :::::\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
Ы
У
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9200095

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         №2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         №2
batchnorm/add_1l
IdentityIdentitybatchnorm/add_1:z:0*
T0*,
_output_shapes
:         №2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         №:::::T P
,
_output_shapes
:         №
 
_user_specified_nameinputs
Ш*
╦
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9201513

inputs
assignmovingavg_9201488
assignmovingavg_1_9201494)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient▒
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                  2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Я
AssignMovingAvg/decayConst**
_class 
loc:@AssignMovingAvg/9201488*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9201488*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp─
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0**
_class 
loc:@AssignMovingAvg/9201488*
_output_shapes
:2
AssignMovingAvg/sub╗
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg/9201488*
_output_shapes
:2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9201488AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/9201488*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/decayConst*,
_class"
 loc:@AssignMovingAvg_1/9201494*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9201494*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╬
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9201494*
_output_shapes
:2
AssignMovingAvg_1/sub┼
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9201494*
_output_shapes
:2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9201494AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/9201494*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  2
batchnorm/add_1┬
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*4
_output_shapes"
 :                  2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                  ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
═
к
7__inference_batch_normalization_1_layer_call_fn_9201723

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ° *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_92002182
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         ° 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         ° ::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ° 
 
_user_specified_nameinputs
Є

*__inference_conv1d_1_layer_call_fn_9201452

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ° *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv1d_1_layer_call_and_return_conditional_losses_92001472
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         ° 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         №::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         №
 
_user_specified_nameinputs
фw
┘	
D__inference_encoder_layer_call_and_return_conditional_losses_9200952
input_16
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource9
5batch_normalization_batchnorm_readvariableop_resource=
9batch_normalization_batchnorm_mul_readvariableop_resource;
7batch_normalization_batchnorm_readvariableop_1_resource;
7batch_normalization_batchnorm_readvariableop_2_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource;
7batch_normalization_1_batchnorm_readvariableop_resource?
;batch_normalization_1_batchnorm_mul_readvariableop_resource=
9batch_normalization_1_batchnorm_readvariableop_1_resource=
9batch_normalization_1_batchnorm_readvariableop_2_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource;
7batch_normalization_2_batchnorm_readvariableop_resource?
;batch_normalization_2_batchnorm_mul_readvariableop_resource=
9batch_normalization_2_batchnorm_readvariableop_1_resource=
9batch_normalization_2_batchnorm_readvariableop_2_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityИЗ
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/conv1d/ExpandDims/dimн
conv1d/conv1d/ExpandDims
ExpandDimsinput_1%conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А2
conv1d/conv1d/ExpandDims═
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOpВ
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim╙
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/conv1d/ExpandDims_1╘
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         №*
paddingVALID*
strides
2
conv1d/conv1dи
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*,
_output_shapes
:         №*
squeeze_dims

¤        2
conv1d/conv1d/Squeezeб
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1d/BiasAdd/ReadVariableOpй
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         №2
conv1d/BiasAddr
conv1d/TanhTanhconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:         №2
conv1d/Tanh╬
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization/batchnorm/ReadVariableOpП
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2%
#batch_normalization/batchnorm/add/y╪
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/addЯ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/Rsqrt┌
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp╒
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/mul└
#batch_normalization/batchnorm/mul_1Mulconv1d/Tanh:y:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:         №2%
#batch_normalization/batchnorm/mul_1╘
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1╒
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/mul_2╘
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2╙
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/sub┌
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:         №2%
#batch_normalization/batchnorm/add_1Л
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_1/conv1d/ExpandDims/dim╙
conv1d_1/conv1d/ExpandDims
ExpandDims'batch_normalization/batchnorm/add_1:z:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         №2
conv1d_1/conv1d/ExpandDims╙
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim█
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_1/conv1d/ExpandDims_1▄
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ° *
paddingVALID*
strides
2
conv1d_1/conv1dо
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*,
_output_shapes
:         ° *
squeeze_dims

¤        2
conv1d_1/conv1d/Squeezeз
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_1/BiasAdd/ReadVariableOp▒
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ° 2
conv1d_1/BiasAddx
conv1d_1/TanhTanhconv1d_1/BiasAdd:output:0*
T0*,
_output_shapes
:         ° 2
conv1d_1/Tanh╘
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_1/batchnorm/ReadVariableOpУ
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_1/batchnorm/add/yр
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_1/batchnorm/addе
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_1/batchnorm/Rsqrtр
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp▌
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_1/batchnorm/mul╚
%batch_normalization_1/batchnorm/mul_1Mulconv1d_1/Tanh:y:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ° 2'
%batch_normalization_1/batchnorm/mul_1┌
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1▌
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_1/batchnorm/mul_2┌
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2█
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_1/batchnorm/subт
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ° 2'
%batch_normalization_1/batchnorm/add_1Л
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_2/conv1d/ExpandDims/dim╒
conv1d_2/conv1d/ExpandDims
ExpandDims)batch_normalization_1/batchnorm/add_1:z:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ° 2
conv1d_2/conv1d/ExpandDims╙
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim█
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d_2/conv1d/ExpandDims_1▄
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         Ї@*
paddingVALID*
strides
2
conv1d_2/conv1dо
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*,
_output_shapes
:         Ї@*
squeeze_dims

¤        2
conv1d_2/conv1d/Squeezeз
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp▒
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         Ї@2
conv1d_2/BiasAddx
conv1d_2/TanhTanhconv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:         Ї@2
conv1d_2/Tanh╘
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOpУ
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_2/batchnorm/add/yр
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/addе
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp▌
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/mul╚
%batch_normalization_2/batchnorm/mul_1Mulconv1d_2/Tanh:y:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:         Ї@2'
%batch_normalization_2/batchnorm/mul_1┌
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1▌
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/mul_2┌
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2█
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/subт
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:         Ї@2'
%batch_normalization_2/batchnorm/add_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     }  2
flatten/Constд
flatten/ReshapeReshape)batch_normalization_2/batchnorm/add_1:z:0flatten/Const:output:0*
T0*)
_output_shapes
:         А·2
flatten/Reshapeб
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
А·*
dtype02
dense/MatMul/ReadVariableOpЧ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/BiasAddj
IdentityIdentitydense/BiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*{
_input_shapesj
h:         А:::::::::::::::::::::U Q
,
_output_shapes
:         А
!
_user_specified_name	input_1
Ъ*
═
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9201923

inputs
assignmovingavg_9201898
assignmovingavg_1_9201904)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient▒
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                  @2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1Я
AssignMovingAvg/decayConst**
_class 
loc:@AssignMovingAvg/9201898*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9201898*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp─
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0**
_class 
loc:@AssignMovingAvg/9201898*
_output_shapes
:@2
AssignMovingAvg/sub╗
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg/9201898*
_output_shapes
:@2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9201898AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/9201898*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/decayConst*,
_class"
 loc:@AssignMovingAvg_1/9201904*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9201904*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╬
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9201904*
_output_shapes
:@2
AssignMovingAvg_1/sub┼
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9201904*
_output_shapes
:@2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9201904AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/9201904*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/add_1┬
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*4
_output_shapes"
 :                  @2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                  @::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
╞
Х
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9201943

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/add_1t
IdentityIdentitybatchnorm/add_1:z:0*
T0*4
_output_shapes"
 :                  @2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                  @:::::\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
╣
Х
)__inference_encoder_layer_call_fn_9201372

inputs
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

unknown_18
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_92006222
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*{
_input_shapesj
h:         А::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
Ш*
╦
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9199680

inputs
assignmovingavg_9199655
assignmovingavg_1_9199661)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradient▒
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                  2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Я
AssignMovingAvg/decayConst**
_class 
loc:@AssignMovingAvg/9199655*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9199655*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp─
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0**
_class 
loc:@AssignMovingAvg/9199655*
_output_shapes
:2
AssignMovingAvg/sub╗
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg/9199655*
_output_shapes
:2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9199655AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/9199655*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/decayConst*,
_class"
 loc:@AssignMovingAvg_1/9199661*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9199661*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╬
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9199661*
_output_shapes
:2
AssignMovingAvg_1/sub┼
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9199661*
_output_shapes
:2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9199661AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/9199661*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  2
batchnorm/add_1┬
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*4
_output_shapes"
 :                  2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                  ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
щ)
═
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9201677

inputs
assignmovingavg_9201652
assignmovingavg_1_9201658)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         ° 2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1Я
AssignMovingAvg/decayConst**
_class 
loc:@AssignMovingAvg/9201652*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9201652*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp─
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0**
_class 
loc:@AssignMovingAvg/9201652*
_output_shapes
: 2
AssignMovingAvg/sub╗
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg/9201652*
_output_shapes
: 2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9201652AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/9201652*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/decayConst*,
_class"
 loc:@AssignMovingAvg_1/9201658*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9201658*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp╬
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9201658*
_output_shapes
: 2
AssignMovingAvg_1/sub┼
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9201658*
_output_shapes
: 2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9201658AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/9201658*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         ° 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         ° 2
batchnorm/add_1║
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*,
_output_shapes
:         ° 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         ° ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:T P
,
_output_shapes
:         ° 
 
_user_specified_nameinputs
З/
║
D__inference_encoder_layer_call_and_return_conditional_losses_9200622

inputs
conv1d_9200573
conv1d_9200575
batch_normalization_9200578
batch_normalization_9200580
batch_normalization_9200582
batch_normalization_9200584
conv1d_1_9200587
conv1d_1_9200589!
batch_normalization_1_9200592!
batch_normalization_1_9200594!
batch_normalization_1_9200596!
batch_normalization_1_9200598
conv1d_2_9200601
conv1d_2_9200603!
batch_normalization_2_9200606!
batch_normalization_2_9200608!
batch_normalization_2_9200610!
batch_normalization_2_9200612
dense_9200616
dense_9200618
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallвconv1d/StatefulPartitionedCallв conv1d_1/StatefulPartitionedCallв conv1d_2/StatefulPartitionedCallвdense/StatefulPartitionedCallТ
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_9200573conv1d_9200575*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         №*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_layer_call_and_return_conditional_losses_92000242 
conv1d/StatefulPartitionedCall▓
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0batch_normalization_9200578batch_normalization_9200580batch_normalization_9200582batch_normalization_9200584*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         №*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_92000952-
+batch_normalization/StatefulPartitionedCall╩
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv1d_1_9200587conv1d_1_9200589*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ° *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv1d_1_layer_call_and_return_conditional_losses_92001472"
 conv1d_1/StatefulPartitionedCall┬
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0batch_normalization_1_9200592batch_normalization_1_9200594batch_normalization_1_9200596batch_normalization_1_9200598*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ° *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_92002182/
-batch_normalization_1/StatefulPartitionedCall╠
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv1d_2_9200601conv1d_2_9200603*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv1d_2_layer_call_and_return_conditional_losses_92002702"
 conv1d_2/StatefulPartitionedCall┬
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_2_9200606batch_normalization_2_9200608batch_normalization_2_9200610batch_normalization_2_9200612*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_92003412/
-batch_normalization_2/StatefulPartitionedCallД
flatten/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         А·* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_92003832
flatten/PartitionedCallв
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_9200616dense_9200618*
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
GPU 2J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_92004012
dense/StatefulPartitionedCallП
IdentityIdentity&dense/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*{
_input_shapesj
h:         А::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
╢
Ц
)__inference_encoder_layer_call_fn_9200997
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

unknown_18
identityИвStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_92005252
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*{
_input_shapesj
h:         А::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:         А
!
_user_specified_name	input_1
Є

*__inference_conv1d_2_layer_call_fn_9201477

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall·
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv1d_2_layer_call_and_return_conditional_losses_92002702
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         Ї@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ° ::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ° 
 
_user_specified_nameinputs
╕
`
D__inference_flatten_layer_call_and_return_conditional_losses_9200383

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     }  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         А·2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         А·2

Identity"
identityIdentity:output:0*+
_input_shapes
:         Ї@:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
─
У
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9201533

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  2
batchnorm/add_1t
IdentityIdentitybatchnorm/add_1:z:0*
T0*4
_output_shapes"
 :                  2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                  :::::\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
╝
Ц
)__inference_encoder_layer_call_fn_9201042
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

unknown_18
identityИвStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_92006222
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*{
_input_shapesj
h:         А::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:         А
!
_user_specified_name	input_1
ч)
╦
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9200075

inputs
assignmovingavg_9200050
assignmovingavg_1_9200056)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         №2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Я
AssignMovingAvg/decayConst**
_class 
loc:@AssignMovingAvg/9200050*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9200050*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp─
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0**
_class 
loc:@AssignMovingAvg/9200050*
_output_shapes
:2
AssignMovingAvg/sub╗
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg/9200050*
_output_shapes
:2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9200050AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/9200050*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/decayConst*,
_class"
 loc:@AssignMovingAvg_1/9200056*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9200056*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╬
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9200056*
_output_shapes
:2
AssignMovingAvg_1/sub┼
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9200056*
_output_shapes
:2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9200056AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/9200056*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         №2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         №2
batchnorm/add_1║
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*,
_output_shapes
:         №2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         №::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:T P
,
_output_shapes
:         №
 
_user_specified_nameinputs
─ф
А
D__inference_encoder_layer_call_and_return_conditional_losses_9201186

inputs6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource/
+batch_normalization_assignmovingavg_92010651
-batch_normalization_assignmovingavg_1_9201071=
9batch_normalization_batchnorm_mul_readvariableop_resource9
5batch_normalization_batchnorm_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource1
-batch_normalization_1_assignmovingavg_92011093
/batch_normalization_1_assignmovingavg_1_9201115?
;batch_normalization_1_batchnorm_mul_readvariableop_resource;
7batch_normalization_1_batchnorm_readvariableop_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource1
-batch_normalization_2_assignmovingavg_92011533
/batch_normalization_2_assignmovingavg_1_9201159?
;batch_normalization_2_batchnorm_mul_readvariableop_resource;
7batch_normalization_2_batchnorm_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityИв7batch_normalization/AssignMovingAvg/AssignSubVariableOpв9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpв9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpв9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpЗ
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/conv1d/ExpandDims/dimм
conv1d/conv1d/ExpandDims
ExpandDimsinputs%conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А2
conv1d/conv1d/ExpandDims═
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOpВ
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim╙
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/conv1d/ExpandDims_1╘
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         №*
paddingVALID*
strides
2
conv1d/conv1dи
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*,
_output_shapes
:         №*
squeeze_dims

¤        2
conv1d/conv1d/Squeezeб
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1d/BiasAdd/ReadVariableOpй
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         №2
conv1d/BiasAddr
conv1d/TanhTanhconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:         №2
conv1d/Tanh╣
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       24
2batch_normalization/moments/mean/reduction_indices╪
 batch_normalization/moments/meanMeanconv1d/Tanh:y:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2"
 batch_normalization/moments/mean╝
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
:2*
(batch_normalization/moments/StopGradientю
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceconv1d/Tanh:y:01batch_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:         №2/
-batch_normalization/moments/SquaredDifference┴
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       28
6batch_normalization/moments/variance/reduction_indicesЖ
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2&
$batch_normalization/moments/variance╜
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze┼
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1█
)batch_normalization/AssignMovingAvg/decayConst*>
_class4
20loc:@batch_normalization/AssignMovingAvg/9201065*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2+
)batch_normalization/AssignMovingAvg/decay╨
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_assignmovingavg_9201065*
_output_shapes
:*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOpи
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg/9201065*
_output_shapes
:2)
'batch_normalization/AssignMovingAvg/subЯ
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg/9201065*
_output_shapes
:2)
'batch_normalization/AssignMovingAvg/mul√
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_assignmovingavg_9201065+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*>
_class4
20loc:@batch_normalization/AssignMovingAvg/9201065*
_output_shapes
 *
dtype029
7batch_normalization/AssignMovingAvg/AssignSubVariableOpс
+batch_normalization/AssignMovingAvg_1/decayConst*@
_class6
42loc:@batch_normalization/AssignMovingAvg_1/9201071*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization/AssignMovingAvg_1/decay╓
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_assignmovingavg_1_9201071*
_output_shapes
:*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp▓
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*@
_class6
42loc:@batch_normalization/AssignMovingAvg_1/9201071*
_output_shapes
:2+
)batch_normalization/AssignMovingAvg_1/subй
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*@
_class6
42loc:@batch_normalization/AssignMovingAvg_1/9201071*
_output_shapes
:2+
)batch_normalization/AssignMovingAvg_1/mulЗ
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_assignmovingavg_1_9201071-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*@
_class6
42loc:@batch_normalization/AssignMovingAvg_1/9201071*
_output_shapes
 *
dtype02;
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpП
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2%
#batch_normalization/batchnorm/add/y╥
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/addЯ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/Rsqrt┌
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp╒
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/mul└
#batch_normalization/batchnorm/mul_1Mulconv1d/Tanh:y:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:         №2%
#batch_normalization/batchnorm/mul_1╦
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/mul_2╬
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization/batchnorm/ReadVariableOp╤
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/sub┌
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:         №2%
#batch_normalization/batchnorm/add_1Л
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_1/conv1d/ExpandDims/dim╙
conv1d_1/conv1d/ExpandDims
ExpandDims'batch_normalization/batchnorm/add_1:z:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         №2
conv1d_1/conv1d/ExpandDims╙
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim█
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_1/conv1d/ExpandDims_1▄
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ° *
paddingVALID*
strides
2
conv1d_1/conv1dо
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*,
_output_shapes
:         ° *
squeeze_dims

¤        2
conv1d_1/conv1d/Squeezeз
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_1/BiasAdd/ReadVariableOp▒
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ° 2
conv1d_1/BiasAddx
conv1d_1/TanhTanhconv1d_1/BiasAdd:output:0*
T0*,
_output_shapes
:         ° 2
conv1d_1/Tanh╜
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indicesр
"batch_normalization_1/moments/meanMeanconv1d_1/Tanh:y:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2$
"batch_normalization_1/moments/mean┬
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
: 2,
*batch_normalization_1/moments/StopGradientЎ
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferenceconv1d_1/Tanh:y:03batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:         ° 21
/batch_normalization_1/moments/SquaredDifference┼
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_1/moments/variance/reduction_indicesО
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2(
&batch_normalization_1/moments/variance├
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze╦
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1с
+batch_normalization_1/AssignMovingAvg/decayConst*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg/9201109*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_1/AssignMovingAvg/decay╓
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_1_assignmovingavg_9201109*
_output_shapes
: *
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp▓
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg/9201109*
_output_shapes
: 2+
)batch_normalization_1/AssignMovingAvg/subй
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg/9201109*
_output_shapes
: 2+
)batch_normalization_1/AssignMovingAvg/mulЗ
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_1_assignmovingavg_9201109-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg/9201109*
_output_shapes
 *
dtype02;
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpч
-batch_normalization_1/AssignMovingAvg_1/decayConst*B
_class8
64loc:@batch_normalization_1/AssignMovingAvg_1/9201115*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_1/AssignMovingAvg_1/decay▄
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_1_assignmovingavg_1_9201115*
_output_shapes
: *
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp╝
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*B
_class8
64loc:@batch_normalization_1/AssignMovingAvg_1/9201115*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg_1/sub│
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_1/AssignMovingAvg_1/9201115*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg_1/mulУ
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_1_assignmovingavg_1_9201115/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_1/AssignMovingAvg_1/9201115*
_output_shapes
 *
dtype02=
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_1/batchnorm/add/y┌
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_1/batchnorm/addе
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_1/batchnorm/Rsqrtр
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp▌
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_1/batchnorm/mul╚
%batch_normalization_1/batchnorm/mul_1Mulconv1d_1/Tanh:y:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ° 2'
%batch_normalization_1/batchnorm/mul_1╙
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_1/batchnorm/mul_2╘
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp┘
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_1/batchnorm/subт
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ° 2'
%batch_normalization_1/batchnorm/add_1Л
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_2/conv1d/ExpandDims/dim╒
conv1d_2/conv1d/ExpandDims
ExpandDims)batch_normalization_1/batchnorm/add_1:z:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ° 2
conv1d_2/conv1d/ExpandDims╙
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim█
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d_2/conv1d/ExpandDims_1▄
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         Ї@*
paddingVALID*
strides
2
conv1d_2/conv1dо
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*,
_output_shapes
:         Ї@*
squeeze_dims

¤        2
conv1d_2/conv1d/Squeezeз
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp▒
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         Ї@2
conv1d_2/BiasAddx
conv1d_2/TanhTanhconv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:         Ї@2
conv1d_2/Tanh╜
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_2/moments/mean/reduction_indicesр
"batch_normalization_2/moments/meanMeanconv1d_2/Tanh:y:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2$
"batch_normalization_2/moments/mean┬
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*"
_output_shapes
:@2,
*batch_normalization_2/moments/StopGradientЎ
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferenceconv1d_2/Tanh:y:03batch_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:         Ї@21
/batch_normalization_2/moments/SquaredDifference┼
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_2/moments/variance/reduction_indicesО
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2(
&batch_normalization_2/moments/variance├
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze╦
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1с
+batch_normalization_2/AssignMovingAvg/decayConst*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg/9201153*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_2/AssignMovingAvg/decay╓
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_2_assignmovingavg_9201153*
_output_shapes
:@*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp▓
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg/9201153*
_output_shapes
:@2+
)batch_normalization_2/AssignMovingAvg/subй
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg/9201153*
_output_shapes
:@2+
)batch_normalization_2/AssignMovingAvg/mulЗ
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_2_assignmovingavg_9201153-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg/9201153*
_output_shapes
 *
dtype02;
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpч
-batch_normalization_2/AssignMovingAvg_1/decayConst*B
_class8
64loc:@batch_normalization_2/AssignMovingAvg_1/9201159*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_2/AssignMovingAvg_1/decay▄
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_2_assignmovingavg_1_9201159*
_output_shapes
:@*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp╝
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*B
_class8
64loc:@batch_normalization_2/AssignMovingAvg_1/9201159*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg_1/sub│
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_2/AssignMovingAvg_1/9201159*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg_1/mulУ
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_2_assignmovingavg_1_9201159/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_2/AssignMovingAvg_1/9201159*
_output_shapes
 *
dtype02=
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_2/batchnorm/add/y┌
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/addе
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp▌
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/mul╚
%batch_normalization_2/batchnorm/mul_1Mulconv1d_2/Tanh:y:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:         Ї@2'
%batch_normalization_2/batchnorm/mul_1╙
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/mul_2╘
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp┘
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/subт
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:         Ї@2'
%batch_normalization_2/batchnorm/add_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     }  2
flatten/Constд
flatten/ReshapeReshape)batch_normalization_2/batchnorm/add_1:z:0flatten/Const:output:0*
T0*)
_output_shapes
:         А·2
flatten/Reshapeб
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
А·*
dtype02
dense/MatMul/ReadVariableOpЧ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/BiasAdd╘
IdentityIdentitydense/BiasAdd:output:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*{
_input_shapesj
h:         А::::::::::::::::::::2r
7batch_normalization/AssignMovingAvg/AssignSubVariableOp7batch_normalization/AssignMovingAvg/AssignSubVariableOp2v
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
╤
к
B__inference_dense_layer_call_and_return_conditional_losses_9200401

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А·*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*0
_input_shapes
:         А·:::Q M
)
_output_shapes
:         А·
 
_user_specified_nameinputs
▄
|
'__inference_dense_layer_call_fn_9201402

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЄ
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
GPU 2J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_92004012
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*0
_input_shapes
:         А·::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:         А·
 
_user_specified_nameinputs
│
Х
)__inference_encoder_layer_call_fn_9201327

inputs
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

unknown_18
identityИвStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *0
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_encoder_layer_call_and_return_conditional_losses_92005252
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*{
_input_shapesj
h:         А::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
Ч
╕
C__inference_conv1d_layer_call_and_return_conditional_losses_9201418

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А2
conv1d/ExpandDims╕
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1╕
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         №*
paddingVALID*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         №*
squeeze_dims

¤        2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         №2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         №2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:         №2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         А:::T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
╞
Х
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9199993

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/add_1t
IdentityIdentitybatchnorm/add_1:z:0*
T0*4
_output_shapes"
 :                  @2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                  @:::::\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
щ4
Ё

 __inference__traced_save_9202052
file_prefix3
/savev2_encoder_dense_kernel_read_readvariableop1
-savev2_encoder_dense_bias_read_readvariableop4
0savev2_encoder_conv1d_kernel_read_readvariableop2
.savev2_encoder_conv1d_bias_read_readvariableop6
2savev2_encoder_conv1d_1_kernel_read_readvariableop4
0savev2_encoder_conv1d_1_bias_read_readvariableop6
2savev2_encoder_conv1d_2_kernel_read_readvariableop4
0savev2_encoder_conv1d_2_bias_read_readvariableop@
<savev2_encoder_batch_normalization_gamma_read_readvariableop?
;savev2_encoder_batch_normalization_beta_read_readvariableopB
>savev2_encoder_batch_normalization_1_gamma_read_readvariableopA
=savev2_encoder_batch_normalization_1_beta_read_readvariableopB
>savev2_encoder_batch_normalization_2_gamma_read_readvariableopA
=savev2_encoder_batch_normalization_2_beta_read_readvariableopF
Bsavev2_encoder_batch_normalization_moving_mean_read_readvariableopJ
Fsavev2_encoder_batch_normalization_moving_variance_read_readvariableopH
Dsavev2_encoder_batch_normalization_1_moving_mean_read_readvariableopL
Hsavev2_encoder_batch_normalization_1_moving_variance_read_readvariableopH
Dsavev2_encoder_batch_normalization_2_moving_mean_read_readvariableopL
Hsavev2_encoder_batch_normalization_2_moving_variance_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
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
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_4ac409d53aa248c29896b4dfae3ec927/part2	
Const_1Л
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
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename╜
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╧
value┼B┬B%out/kernel/.ATTRIBUTES/VARIABLE_VALUEB#out/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names▓
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesБ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_encoder_dense_kernel_read_readvariableop-savev2_encoder_dense_bias_read_readvariableop0savev2_encoder_conv1d_kernel_read_readvariableop.savev2_encoder_conv1d_bias_read_readvariableop2savev2_encoder_conv1d_1_kernel_read_readvariableop0savev2_encoder_conv1d_1_bias_read_readvariableop2savev2_encoder_conv1d_2_kernel_read_readvariableop0savev2_encoder_conv1d_2_bias_read_readvariableop<savev2_encoder_batch_normalization_gamma_read_readvariableop;savev2_encoder_batch_normalization_beta_read_readvariableop>savev2_encoder_batch_normalization_1_gamma_read_readvariableop=savev2_encoder_batch_normalization_1_beta_read_readvariableop>savev2_encoder_batch_normalization_2_gamma_read_readvariableop=savev2_encoder_batch_normalization_2_beta_read_readvariableopBsavev2_encoder_batch_normalization_moving_mean_read_readvariableopFsavev2_encoder_batch_normalization_moving_variance_read_readvariableopDsavev2_encoder_batch_normalization_1_moving_mean_read_readvariableopHsavev2_encoder_batch_normalization_1_moving_variance_read_readvariableopDsavev2_encoder_batch_normalization_2_moving_mean_read_readvariableopHsavev2_encoder_batch_normalization_2_moving_variance_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *#
dtypes
22
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
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

identity_1Identity_1:output:0*п
_input_shapesЭ
Ъ: :
А·:::: : : @:@::: : :@:@::: : :@:@: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
А·: 

_output_shapes
::($
"
_output_shapes
:: 

_output_shapes
::($
"
_output_shapes
: : 

_output_shapes
: :($
"
_output_shapes
: @: 

_output_shapes
:@: 	

_output_shapes
:: 


_output_shapes
:: 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
: : 

_output_shapes
: : 

_output_shapes
:@: 

_output_shapes
:@:

_output_shapes
: 
щ
и
5__inference_batch_normalization_layer_call_fn_9201559

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_91997132
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :                  2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                  ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
ы
к
7__inference_batch_normalization_1_layer_call_fn_9201792

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_91998202
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :                   2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                   ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
Ч
╕
C__inference_conv1d_layer_call_and_return_conditional_losses_9200024

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А2
conv1d/ExpandDims╕
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/ExpandDims_1╕
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         №*
paddingVALID*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         №*
squeeze_dims

¤        2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         №2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         №2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:         №2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         А:::T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
Ъ*
═
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9199820

inputs
assignmovingavg_9199795
assignmovingavg_1_9199801)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient▒
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                   2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1Я
AssignMovingAvg/decayConst**
_class 
loc:@AssignMovingAvg/9199795*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9199795*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp─
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0**
_class 
loc:@AssignMovingAvg/9199795*
_output_shapes
: 2
AssignMovingAvg/sub╗
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg/9199795*
_output_shapes
: 2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9199795AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/9199795*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/decayConst*,
_class"
 loc:@AssignMovingAvg_1/9199801*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9199801*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp╬
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9199801*
_output_shapes
: 2
AssignMovingAvg_1/sub┼
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9199801*
_output_shapes
: 2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9199801AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/9199801*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                   2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                   2
batchnorm/add_1┬
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*4
_output_shapes"
 :                   2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                   ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
Э
Х
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9201697

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         ° 2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         ° 2
batchnorm/add_1l
IdentityIdentitybatchnorm/add_1:z:0*
T0*,
_output_shapes
:         ° 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         ° :::::T P
,
_output_shapes
:         ° 
 
_user_specified_nameinputs
─
У
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9199713

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  2
batchnorm/add_1t
IdentityIdentitybatchnorm/add_1:z:0*
T0*4
_output_shapes"
 :                  2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                  :::::\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Щ
║
E__inference_conv1d_1_layer_call_and_return_conditional_losses_9201443

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         №2
conv1d/ExpandDims╕
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1╕
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ° *
paddingVALID*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ° *
squeeze_dims

¤        2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ° 2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ° 2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:         ° 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         №:::T P
,
_output_shapes
:         №
 
_user_specified_nameinputs
сw
╪	
D__inference_encoder_layer_call_and_return_conditional_losses_9201282

inputs6
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource9
5batch_normalization_batchnorm_readvariableop_resource=
9batch_normalization_batchnorm_mul_readvariableop_resource;
7batch_normalization_batchnorm_readvariableop_1_resource;
7batch_normalization_batchnorm_readvariableop_2_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource;
7batch_normalization_1_batchnorm_readvariableop_resource?
;batch_normalization_1_batchnorm_mul_readvariableop_resource=
9batch_normalization_1_batchnorm_readvariableop_1_resource=
9batch_normalization_1_batchnorm_readvariableop_2_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource;
7batch_normalization_2_batchnorm_readvariableop_resource?
;batch_normalization_2_batchnorm_mul_readvariableop_resource=
9batch_normalization_2_batchnorm_readvariableop_1_resource=
9batch_normalization_2_batchnorm_readvariableop_2_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityИЗ
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/conv1d/ExpandDims/dimм
conv1d/conv1d/ExpandDims
ExpandDimsinputs%conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А2
conv1d/conv1d/ExpandDims═
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOpВ
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim╙
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/conv1d/ExpandDims_1╘
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         №*
paddingVALID*
strides
2
conv1d/conv1dи
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*,
_output_shapes
:         №*
squeeze_dims

¤        2
conv1d/conv1d/Squeezeб
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1d/BiasAdd/ReadVariableOpй
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         №2
conv1d/BiasAddr
conv1d/TanhTanhconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:         №2
conv1d/Tanh╬
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization/batchnorm/ReadVariableOpП
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2%
#batch_normalization/batchnorm/add/y╪
!batch_normalization/batchnorm/addAddV24batch_normalization/batchnorm/ReadVariableOp:value:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/addЯ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/Rsqrt┌
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp╒
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/mul└
#batch_normalization/batchnorm/mul_1Mulconv1d/Tanh:y:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:         №2%
#batch_normalization/batchnorm/mul_1╘
.batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp7batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_1╒
#batch_normalization/batchnorm/mul_2Mul6batch_normalization/batchnorm/ReadVariableOp_1:value:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/mul_2╘
.batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp7batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype020
.batch_normalization/batchnorm/ReadVariableOp_2╙
!batch_normalization/batchnorm/subSub6batch_normalization/batchnorm/ReadVariableOp_2:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/sub┌
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:         №2%
#batch_normalization/batchnorm/add_1Л
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_1/conv1d/ExpandDims/dim╙
conv1d_1/conv1d/ExpandDims
ExpandDims'batch_normalization/batchnorm/add_1:z:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         №2
conv1d_1/conv1d/ExpandDims╙
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim█
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_1/conv1d/ExpandDims_1▄
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ° *
paddingVALID*
strides
2
conv1d_1/conv1dо
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*,
_output_shapes
:         ° *
squeeze_dims

¤        2
conv1d_1/conv1d/Squeezeз
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_1/BiasAdd/ReadVariableOp▒
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ° 2
conv1d_1/BiasAddx
conv1d_1/TanhTanhconv1d_1/BiasAdd:output:0*
T0*,
_output_shapes
:         ° 2
conv1d_1/Tanh╘
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_1/batchnorm/ReadVariableOpУ
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_1/batchnorm/add/yр
#batch_normalization_1/batchnorm/addAddV26batch_normalization_1/batchnorm/ReadVariableOp:value:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_1/batchnorm/addе
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_1/batchnorm/Rsqrtр
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp▌
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_1/batchnorm/mul╚
%batch_normalization_1/batchnorm/mul_1Mulconv1d_1/Tanh:y:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ° 2'
%batch_normalization_1/batchnorm/mul_1┌
0batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_1▌
%batch_normalization_1/batchnorm/mul_2Mul8batch_normalization_1/batchnorm/ReadVariableOp_1:value:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_1/batchnorm/mul_2┌
0batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype022
0batch_normalization_1/batchnorm/ReadVariableOp_2█
#batch_normalization_1/batchnorm/subSub8batch_normalization_1/batchnorm/ReadVariableOp_2:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_1/batchnorm/subт
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ° 2'
%batch_normalization_1/batchnorm/add_1Л
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_2/conv1d/ExpandDims/dim╒
conv1d_2/conv1d/ExpandDims
ExpandDims)batch_normalization_1/batchnorm/add_1:z:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ° 2
conv1d_2/conv1d/ExpandDims╙
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim█
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d_2/conv1d/ExpandDims_1▄
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         Ї@*
paddingVALID*
strides
2
conv1d_2/conv1dо
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*,
_output_shapes
:         Ї@*
squeeze_dims

¤        2
conv1d_2/conv1d/Squeezeз
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp▒
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         Ї@2
conv1d_2/BiasAddx
conv1d_2/TanhTanhconv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:         Ї@2
conv1d_2/Tanh╘
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOpУ
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_2/batchnorm/add/yр
#batch_normalization_2/batchnorm/addAddV26batch_normalization_2/batchnorm/ReadVariableOp:value:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/addе
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp▌
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/mul╚
%batch_normalization_2/batchnorm/mul_1Mulconv1d_2/Tanh:y:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:         Ї@2'
%batch_normalization_2/batchnorm/mul_1┌
0batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_1▌
%batch_normalization_2/batchnorm/mul_2Mul8batch_normalization_2/batchnorm/ReadVariableOp_1:value:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/mul_2┌
0batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOp9batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype022
0batch_normalization_2/batchnorm/ReadVariableOp_2█
#batch_normalization_2/batchnorm/subSub8batch_normalization_2/batchnorm/ReadVariableOp_2:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/subт
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:         Ї@2'
%batch_normalization_2/batchnorm/add_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     }  2
flatten/Constд
flatten/ReshapeReshape)batch_normalization_2/batchnorm/add_1:z:0flatten/Const:output:0*
T0*)
_output_shapes
:         А·2
flatten/Reshapeб
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
А·*
dtype02
dense/MatMul/ReadVariableOpЧ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/BiasAddj
IdentityIdentitydense/BiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*{
_input_shapesj
h:         А:::::::::::::::::::::T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
Э
Х
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9200218

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         ° 2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         ° 2
batchnorm/add_1l
IdentityIdentitybatchnorm/add_1:z:0*
T0*,
_output_shapes
:         ° 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         ° :::::T P
,
_output_shapes
:         ° 
 
_user_specified_nameinputs
Ы
У
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9201615

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         №2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         №2
batchnorm/add_1l
IdentityIdentitybatchnorm/add_1:z:0*
T0*,
_output_shapes
:         №2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         №:::::T P
,
_output_shapes
:         №
 
_user_specified_nameinputs
ы
к
7__inference_batch_normalization_2_layer_call_fn_9201956

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallз
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_91999602
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :                  @2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                  @::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
Ц
Т
%__inference_signature_wrapper_9200712
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

unknown_18
identityИвStatefulPartitionedCall╞
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference__wrapped_model_91995842
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*{
_input_shapesj
h:         А::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:         А
!
_user_specified_name	input_1
╟ф
Б
D__inference_encoder_layer_call_and_return_conditional_losses_9200856
input_16
2conv1d_conv1d_expanddims_1_readvariableop_resource*
&conv1d_biasadd_readvariableop_resource/
+batch_normalization_assignmovingavg_92007351
-batch_normalization_assignmovingavg_1_9200741=
9batch_normalization_batchnorm_mul_readvariableop_resource9
5batch_normalization_batchnorm_readvariableop_resource8
4conv1d_1_conv1d_expanddims_1_readvariableop_resource,
(conv1d_1_biasadd_readvariableop_resource1
-batch_normalization_1_assignmovingavg_92007793
/batch_normalization_1_assignmovingavg_1_9200785?
;batch_normalization_1_batchnorm_mul_readvariableop_resource;
7batch_normalization_1_batchnorm_readvariableop_resource8
4conv1d_2_conv1d_expanddims_1_readvariableop_resource,
(conv1d_2_biasadd_readvariableop_resource1
-batch_normalization_2_assignmovingavg_92008233
/batch_normalization_2_assignmovingavg_1_9200829?
;batch_normalization_2_batchnorm_mul_readvariableop_resource;
7batch_normalization_2_batchnorm_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityИв7batch_normalization/AssignMovingAvg/AssignSubVariableOpв9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpв9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpв9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpв;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpЗ
conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/conv1d/ExpandDims/dimн
conv1d/conv1d/ExpandDims
ExpandDimsinput_1%conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А2
conv1d/conv1d/ExpandDims═
)conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp2conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype02+
)conv1d/conv1d/ExpandDims_1/ReadVariableOpВ
conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2 
conv1d/conv1d/ExpandDims_1/dim╙
conv1d/conv1d/ExpandDims_1
ExpandDims1conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0'conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2
conv1d/conv1d/ExpandDims_1╘
conv1d/conv1dConv2D!conv1d/conv1d/ExpandDims:output:0#conv1d/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         №*
paddingVALID*
strides
2
conv1d/conv1dи
conv1d/conv1d/SqueezeSqueezeconv1d/conv1d:output:0*
T0*,
_output_shapes
:         №*
squeeze_dims

¤        2
conv1d/conv1d/Squeezeб
conv1d/BiasAdd/ReadVariableOpReadVariableOp&conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
conv1d/BiasAdd/ReadVariableOpй
conv1d/BiasAddBiasAddconv1d/conv1d/Squeeze:output:0%conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         №2
conv1d/BiasAddr
conv1d/TanhTanhconv1d/BiasAdd:output:0*
T0*,
_output_shapes
:         №2
conv1d/Tanh╣
2batch_normalization/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       24
2batch_normalization/moments/mean/reduction_indices╪
 batch_normalization/moments/meanMeanconv1d/Tanh:y:0;batch_normalization/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2"
 batch_normalization/moments/mean╝
(batch_normalization/moments/StopGradientStopGradient)batch_normalization/moments/mean:output:0*
T0*"
_output_shapes
:2*
(batch_normalization/moments/StopGradientю
-batch_normalization/moments/SquaredDifferenceSquaredDifferenceconv1d/Tanh:y:01batch_normalization/moments/StopGradient:output:0*
T0*,
_output_shapes
:         №2/
-batch_normalization/moments/SquaredDifference┴
6batch_normalization/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       28
6batch_normalization/moments/variance/reduction_indicesЖ
$batch_normalization/moments/varianceMean1batch_normalization/moments/SquaredDifference:z:0?batch_normalization/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2&
$batch_normalization/moments/variance╜
#batch_normalization/moments/SqueezeSqueeze)batch_normalization/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2%
#batch_normalization/moments/Squeeze┼
%batch_normalization/moments/Squeeze_1Squeeze-batch_normalization/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2'
%batch_normalization/moments/Squeeze_1█
)batch_normalization/AssignMovingAvg/decayConst*>
_class4
20loc:@batch_normalization/AssignMovingAvg/9200735*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2+
)batch_normalization/AssignMovingAvg/decay╨
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_assignmovingavg_9200735*
_output_shapes
:*
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOpи
'batch_normalization/AssignMovingAvg/subSub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:0,batch_normalization/moments/Squeeze:output:0*
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg/9200735*
_output_shapes
:2)
'batch_normalization/AssignMovingAvg/subЯ
'batch_normalization/AssignMovingAvg/mulMul+batch_normalization/AssignMovingAvg/sub:z:02batch_normalization/AssignMovingAvg/decay:output:0*
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg/9200735*
_output_shapes
:2)
'batch_normalization/AssignMovingAvg/mul√
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_assignmovingavg_9200735+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*>
_class4
20loc:@batch_normalization/AssignMovingAvg/9200735*
_output_shapes
 *
dtype029
7batch_normalization/AssignMovingAvg/AssignSubVariableOpс
+batch_normalization/AssignMovingAvg_1/decayConst*@
_class6
42loc:@batch_normalization/AssignMovingAvg_1/9200741*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization/AssignMovingAvg_1/decay╓
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_assignmovingavg_1_9200741*
_output_shapes
:*
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp▓
)batch_normalization/AssignMovingAvg_1/subSub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:0.batch_normalization/moments/Squeeze_1:output:0*
T0*@
_class6
42loc:@batch_normalization/AssignMovingAvg_1/9200741*
_output_shapes
:2+
)batch_normalization/AssignMovingAvg_1/subй
)batch_normalization/AssignMovingAvg_1/mulMul-batch_normalization/AssignMovingAvg_1/sub:z:04batch_normalization/AssignMovingAvg_1/decay:output:0*
T0*@
_class6
42loc:@batch_normalization/AssignMovingAvg_1/9200741*
_output_shapes
:2+
)batch_normalization/AssignMovingAvg_1/mulЗ
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_assignmovingavg_1_9200741-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*@
_class6
42loc:@batch_normalization/AssignMovingAvg_1/9200741*
_output_shapes
 *
dtype02;
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpП
#batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2%
#batch_normalization/batchnorm/add/y╥
!batch_normalization/batchnorm/addAddV2.batch_normalization/moments/Squeeze_1:output:0,batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/addЯ
#batch_normalization/batchnorm/RsqrtRsqrt%batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/Rsqrt┌
0batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOp9batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype022
0batch_normalization/batchnorm/mul/ReadVariableOp╒
!batch_normalization/batchnorm/mulMul'batch_normalization/batchnorm/Rsqrt:y:08batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/mul└
#batch_normalization/batchnorm/mul_1Mulconv1d/Tanh:y:0%batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:         №2%
#batch_normalization/batchnorm/mul_1╦
#batch_normalization/batchnorm/mul_2Mul,batch_normalization/moments/Squeeze:output:0%batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2%
#batch_normalization/batchnorm/mul_2╬
,batch_normalization/batchnorm/ReadVariableOpReadVariableOp5batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02.
,batch_normalization/batchnorm/ReadVariableOp╤
!batch_normalization/batchnorm/subSub4batch_normalization/batchnorm/ReadVariableOp:value:0'batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2#
!batch_normalization/batchnorm/sub┌
#batch_normalization/batchnorm/add_1AddV2'batch_normalization/batchnorm/mul_1:z:0%batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:         №2%
#batch_normalization/batchnorm/add_1Л
conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_1/conv1d/ExpandDims/dim╙
conv1d_1/conv1d/ExpandDims
ExpandDims'batch_normalization/batchnorm/add_1:z:0'conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         №2
conv1d_1/conv1d/ExpandDims╙
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02-
+conv1d_1/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_1/conv1d/ExpandDims_1/dim█
conv1d_1/conv1d/ExpandDims_1
ExpandDims3conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d_1/conv1d/ExpandDims_1▄
conv1d_1/conv1dConv2D#conv1d_1/conv1d/ExpandDims:output:0%conv1d_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ° *
paddingVALID*
strides
2
conv1d_1/conv1dо
conv1d_1/conv1d/SqueezeSqueezeconv1d_1/conv1d:output:0*
T0*,
_output_shapes
:         ° *
squeeze_dims

¤        2
conv1d_1/conv1d/Squeezeз
conv1d_1/BiasAdd/ReadVariableOpReadVariableOp(conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv1d_1/BiasAdd/ReadVariableOp▒
conv1d_1/BiasAddBiasAdd conv1d_1/conv1d/Squeeze:output:0'conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ° 2
conv1d_1/BiasAddx
conv1d_1/TanhTanhconv1d_1/BiasAdd:output:0*
T0*,
_output_shapes
:         ° 2
conv1d_1/Tanh╜
4batch_normalization_1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_1/moments/mean/reduction_indicesр
"batch_normalization_1/moments/meanMeanconv1d_1/Tanh:y:0=batch_normalization_1/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2$
"batch_normalization_1/moments/mean┬
*batch_normalization_1/moments/StopGradientStopGradient+batch_normalization_1/moments/mean:output:0*
T0*"
_output_shapes
: 2,
*batch_normalization_1/moments/StopGradientЎ
/batch_normalization_1/moments/SquaredDifferenceSquaredDifferenceconv1d_1/Tanh:y:03batch_normalization_1/moments/StopGradient:output:0*
T0*,
_output_shapes
:         ° 21
/batch_normalization_1/moments/SquaredDifference┼
8batch_normalization_1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_1/moments/variance/reduction_indicesО
&batch_normalization_1/moments/varianceMean3batch_normalization_1/moments/SquaredDifference:z:0Abatch_normalization_1/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2(
&batch_normalization_1/moments/variance├
%batch_normalization_1/moments/SqueezeSqueeze+batch_normalization_1/moments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2'
%batch_normalization_1/moments/Squeeze╦
'batch_normalization_1/moments/Squeeze_1Squeeze/batch_normalization_1/moments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2)
'batch_normalization_1/moments/Squeeze_1с
+batch_normalization_1/AssignMovingAvg/decayConst*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg/9200779*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_1/AssignMovingAvg/decay╓
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_1_assignmovingavg_9200779*
_output_shapes
: *
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp▓
)batch_normalization_1/AssignMovingAvg/subSub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_1/moments/Squeeze:output:0*
T0*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg/9200779*
_output_shapes
: 2+
)batch_normalization_1/AssignMovingAvg/subй
)batch_normalization_1/AssignMovingAvg/mulMul-batch_normalization_1/AssignMovingAvg/sub:z:04batch_normalization_1/AssignMovingAvg/decay:output:0*
T0*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg/9200779*
_output_shapes
: 2+
)batch_normalization_1/AssignMovingAvg/mulЗ
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_1_assignmovingavg_9200779-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg/9200779*
_output_shapes
 *
dtype02;
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpч
-batch_normalization_1/AssignMovingAvg_1/decayConst*B
_class8
64loc:@batch_normalization_1/AssignMovingAvg_1/9200785*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_1/AssignMovingAvg_1/decay▄
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_1_assignmovingavg_1_9200785*
_output_shapes
: *
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp╝
+batch_normalization_1/AssignMovingAvg_1/subSub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_1/moments/Squeeze_1:output:0*
T0*B
_class8
64loc:@batch_normalization_1/AssignMovingAvg_1/9200785*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg_1/sub│
+batch_normalization_1/AssignMovingAvg_1/mulMul/batch_normalization_1/AssignMovingAvg_1/sub:z:06batch_normalization_1/AssignMovingAvg_1/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_1/AssignMovingAvg_1/9200785*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg_1/mulУ
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_1_assignmovingavg_1_9200785/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_1/AssignMovingAvg_1/9200785*
_output_shapes
 *
dtype02=
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_1/batchnorm/add/y┌
#batch_normalization_1/batchnorm/addAddV20batch_normalization_1/moments/Squeeze_1:output:0.batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2%
#batch_normalization_1/batchnorm/addе
%batch_normalization_1/batchnorm/RsqrtRsqrt'batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_1/batchnorm/Rsqrtр
2batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype024
2batch_normalization_1/batchnorm/mul/ReadVariableOp▌
#batch_normalization_1/batchnorm/mulMul)batch_normalization_1/batchnorm/Rsqrt:y:0:batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2%
#batch_normalization_1/batchnorm/mul╚
%batch_normalization_1/batchnorm/mul_1Mulconv1d_1/Tanh:y:0'batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ° 2'
%batch_normalization_1/batchnorm/mul_1╙
%batch_normalization_1/batchnorm/mul_2Mul.batch_normalization_1/moments/Squeeze:output:0'batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
: 2'
%batch_normalization_1/batchnorm/mul_2╘
.batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype020
.batch_normalization_1/batchnorm/ReadVariableOp┘
#batch_normalization_1/batchnorm/subSub6batch_normalization_1/batchnorm/ReadVariableOp:value:0)batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2%
#batch_normalization_1/batchnorm/subт
%batch_normalization_1/batchnorm/add_1AddV2)batch_normalization_1/batchnorm/mul_1:z:0'batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ° 2'
%batch_normalization_1/batchnorm/add_1Л
conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2 
conv1d_2/conv1d/ExpandDims/dim╒
conv1d_2/conv1d/ExpandDims
ExpandDims)batch_normalization_1/batchnorm/add_1:z:0'conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ° 2
conv1d_2/conv1d/ExpandDims╙
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp4conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02-
+conv1d_2/conv1d/ExpandDims_1/ReadVariableOpЖ
 conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 conv1d_2/conv1d/ExpandDims_1/dim█
conv1d_2/conv1d/ExpandDims_1
ExpandDims3conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:0)conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d_2/conv1d/ExpandDims_1▄
conv1d_2/conv1dConv2D#conv1d_2/conv1d/ExpandDims:output:0%conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         Ї@*
paddingVALID*
strides
2
conv1d_2/conv1dо
conv1d_2/conv1d/SqueezeSqueezeconv1d_2/conv1d:output:0*
T0*,
_output_shapes
:         Ї@*
squeeze_dims

¤        2
conv1d_2/conv1d/Squeezeз
conv1d_2/BiasAdd/ReadVariableOpReadVariableOp(conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv1d_2/BiasAdd/ReadVariableOp▒
conv1d_2/BiasAddBiasAdd conv1d_2/conv1d/Squeeze:output:0'conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         Ї@2
conv1d_2/BiasAddx
conv1d_2/TanhTanhconv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:         Ї@2
conv1d_2/Tanh╜
4batch_normalization_2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       26
4batch_normalization_2/moments/mean/reduction_indicesр
"batch_normalization_2/moments/meanMeanconv1d_2/Tanh:y:0=batch_normalization_2/moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2$
"batch_normalization_2/moments/mean┬
*batch_normalization_2/moments/StopGradientStopGradient+batch_normalization_2/moments/mean:output:0*
T0*"
_output_shapes
:@2,
*batch_normalization_2/moments/StopGradientЎ
/batch_normalization_2/moments/SquaredDifferenceSquaredDifferenceconv1d_2/Tanh:y:03batch_normalization_2/moments/StopGradient:output:0*
T0*,
_output_shapes
:         Ї@21
/batch_normalization_2/moments/SquaredDifference┼
8batch_normalization_2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2:
8batch_normalization_2/moments/variance/reduction_indicesО
&batch_normalization_2/moments/varianceMean3batch_normalization_2/moments/SquaredDifference:z:0Abatch_normalization_2/moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2(
&batch_normalization_2/moments/variance├
%batch_normalization_2/moments/SqueezeSqueeze+batch_normalization_2/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2'
%batch_normalization_2/moments/Squeeze╦
'batch_normalization_2/moments/Squeeze_1Squeeze/batch_normalization_2/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2)
'batch_normalization_2/moments/Squeeze_1с
+batch_normalization_2/AssignMovingAvg/decayConst*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg/9200823*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2-
+batch_normalization_2/AssignMovingAvg/decay╓
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp-batch_normalization_2_assignmovingavg_9200823*
_output_shapes
:@*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp▓
)batch_normalization_2/AssignMovingAvg/subSub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:0.batch_normalization_2/moments/Squeeze:output:0*
T0*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg/9200823*
_output_shapes
:@2+
)batch_normalization_2/AssignMovingAvg/subй
)batch_normalization_2/AssignMovingAvg/mulMul-batch_normalization_2/AssignMovingAvg/sub:z:04batch_normalization_2/AssignMovingAvg/decay:output:0*
T0*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg/9200823*
_output_shapes
:@2+
)batch_normalization_2/AssignMovingAvg/mulЗ
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp-batch_normalization_2_assignmovingavg_9200823-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg/9200823*
_output_shapes
 *
dtype02;
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpч
-batch_normalization_2/AssignMovingAvg_1/decayConst*B
_class8
64loc:@batch_normalization_2/AssignMovingAvg_1/9200829*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2/
-batch_normalization_2/AssignMovingAvg_1/decay▄
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp/batch_normalization_2_assignmovingavg_1_9200829*
_output_shapes
:@*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp╝
+batch_normalization_2/AssignMovingAvg_1/subSub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:00batch_normalization_2/moments/Squeeze_1:output:0*
T0*B
_class8
64loc:@batch_normalization_2/AssignMovingAvg_1/9200829*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg_1/sub│
+batch_normalization_2/AssignMovingAvg_1/mulMul/batch_normalization_2/AssignMovingAvg_1/sub:z:06batch_normalization_2/AssignMovingAvg_1/decay:output:0*
T0*B
_class8
64loc:@batch_normalization_2/AssignMovingAvg_1/9200829*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg_1/mulУ
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp/batch_normalization_2_assignmovingavg_1_9200829/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*B
_class8
64loc:@batch_normalization_2/AssignMovingAvg_1/9200829*
_output_shapes
 *
dtype02=
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpУ
%batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2'
%batch_normalization_2/batchnorm/add/y┌
#batch_normalization_2/batchnorm/addAddV20batch_normalization_2/moments/Squeeze_1:output:0.batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/addе
%batch_normalization_2/batchnorm/RsqrtRsqrt'batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/Rsqrtр
2batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOp;batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype024
2batch_normalization_2/batchnorm/mul/ReadVariableOp▌
#batch_normalization_2/batchnorm/mulMul)batch_normalization_2/batchnorm/Rsqrt:y:0:batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/mul╚
%batch_normalization_2/batchnorm/mul_1Mulconv1d_2/Tanh:y:0'batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:         Ї@2'
%batch_normalization_2/batchnorm/mul_1╙
%batch_normalization_2/batchnorm/mul_2Mul.batch_normalization_2/moments/Squeeze:output:0'batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@2'
%batch_normalization_2/batchnorm/mul_2╘
.batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp7batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype020
.batch_normalization_2/batchnorm/ReadVariableOp┘
#batch_normalization_2/batchnorm/subSub6batch_normalization_2/batchnorm/ReadVariableOp:value:0)batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2%
#batch_normalization_2/batchnorm/subт
%batch_normalization_2/batchnorm/add_1AddV2)batch_normalization_2/batchnorm/mul_1:z:0'batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:         Ї@2'
%batch_normalization_2/batchnorm/add_1o
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     }  2
flatten/Constд
flatten/ReshapeReshape)batch_normalization_2/batchnorm/add_1:z:0flatten/Const:output:0*
T0*)
_output_shapes
:         А·2
flatten/Reshapeб
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
А·*
dtype02
dense/MatMul/ReadVariableOpЧ
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense/BiasAdd╘
IdentityIdentitydense/BiasAdd:output:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*{
_input_shapesj
h:         А::::::::::::::::::::2r
7batch_normalization/AssignMovingAvg/AssignSubVariableOp7batch_normalization/AssignMovingAvg/AssignSubVariableOp2v
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp2v
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp2z
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp:U Q
,
_output_shapes
:         А
!
_user_specified_name	input_1
Э
Х
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9201861

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         Ї@2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         Ї@2
batchnorm/add_1l
IdentityIdentitybatchnorm/add_1:z:0*
T0*,
_output_shapes
:         Ї@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         Ї@:::::T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
╬X
с
#__inference__traced_restore_9202122
file_prefix)
%assignvariableop_encoder_dense_kernel)
%assignvariableop_1_encoder_dense_bias,
(assignvariableop_2_encoder_conv1d_kernel*
&assignvariableop_3_encoder_conv1d_bias.
*assignvariableop_4_encoder_conv1d_1_kernel,
(assignvariableop_5_encoder_conv1d_1_bias.
*assignvariableop_6_encoder_conv1d_2_kernel,
(assignvariableop_7_encoder_conv1d_2_bias8
4assignvariableop_8_encoder_batch_normalization_gamma7
3assignvariableop_9_encoder_batch_normalization_beta;
7assignvariableop_10_encoder_batch_normalization_1_gamma:
6assignvariableop_11_encoder_batch_normalization_1_beta;
7assignvariableop_12_encoder_batch_normalization_2_gamma:
6assignvariableop_13_encoder_batch_normalization_2_beta?
;assignvariableop_14_encoder_batch_normalization_moving_meanC
?assignvariableop_15_encoder_batch_normalization_moving_varianceA
=assignvariableop_16_encoder_batch_normalization_1_moving_meanE
Aassignvariableop_17_encoder_batch_normalization_1_moving_varianceA
=assignvariableop_18_encoder_batch_normalization_2_moving_meanE
Aassignvariableop_19_encoder_batch_normalization_2_moving_variance
identity_21ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9├
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*╧
value┼B┬B%out/kernel/.ATTRIBUTES/VARIABLE_VALUEB#out/bias/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names╕
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesФ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityд
AssignVariableOpAssignVariableOp%assignvariableop_encoder_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1к
AssignVariableOp_1AssignVariableOp%assignvariableop_1_encoder_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2н
AssignVariableOp_2AssignVariableOp(assignvariableop_2_encoder_conv1d_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3л
AssignVariableOp_3AssignVariableOp&assignvariableop_3_encoder_conv1d_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4п
AssignVariableOp_4AssignVariableOp*assignvariableop_4_encoder_conv1d_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5н
AssignVariableOp_5AssignVariableOp(assignvariableop_5_encoder_conv1d_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6п
AssignVariableOp_6AssignVariableOp*assignvariableop_6_encoder_conv1d_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7н
AssignVariableOp_7AssignVariableOp(assignvariableop_7_encoder_conv1d_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8╣
AssignVariableOp_8AssignVariableOp4assignvariableop_8_encoder_batch_normalization_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9╕
AssignVariableOp_9AssignVariableOp3assignvariableop_9_encoder_batch_normalization_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10┐
AssignVariableOp_10AssignVariableOp7assignvariableop_10_encoder_batch_normalization_1_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11╛
AssignVariableOp_11AssignVariableOp6assignvariableop_11_encoder_batch_normalization_1_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12┐
AssignVariableOp_12AssignVariableOp7assignvariableop_12_encoder_batch_normalization_2_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13╛
AssignVariableOp_13AssignVariableOp6assignvariableop_13_encoder_batch_normalization_2_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14├
AssignVariableOp_14AssignVariableOp;assignvariableop_14_encoder_batch_normalization_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15╟
AssignVariableOp_15AssignVariableOp?assignvariableop_15_encoder_batch_normalization_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16┼
AssignVariableOp_16AssignVariableOp=assignvariableop_16_encoder_batch_normalization_1_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17╔
AssignVariableOp_17AssignVariableOpAassignvariableop_17_encoder_batch_normalization_1_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18┼
AssignVariableOp_18AssignVariableOp=assignvariableop_18_encoder_batch_normalization_2_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19╔
AssignVariableOp_19AssignVariableOpAassignvariableop_19_encoder_batch_normalization_2_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_199
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЦ
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_20Й
Identity_21IdentityIdentity_20:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_21"#
identity_21Identity_21:output:0*e
_input_shapesT
R: ::::::::::::::::::::2$
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
▌З
╫

"__inference__wrapped_model_9199584
input_1>
:encoder_conv1d_conv1d_expanddims_1_readvariableop_resource2
.encoder_conv1d_biasadd_readvariableop_resourceA
=encoder_batch_normalization_batchnorm_readvariableop_resourceE
Aencoder_batch_normalization_batchnorm_mul_readvariableop_resourceC
?encoder_batch_normalization_batchnorm_readvariableop_1_resourceC
?encoder_batch_normalization_batchnorm_readvariableop_2_resource@
<encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource4
0encoder_conv1d_1_biasadd_readvariableop_resourceC
?encoder_batch_normalization_1_batchnorm_readvariableop_resourceG
Cencoder_batch_normalization_1_batchnorm_mul_readvariableop_resourceE
Aencoder_batch_normalization_1_batchnorm_readvariableop_1_resourceE
Aencoder_batch_normalization_1_batchnorm_readvariableop_2_resource@
<encoder_conv1d_2_conv1d_expanddims_1_readvariableop_resource4
0encoder_conv1d_2_biasadd_readvariableop_resourceC
?encoder_batch_normalization_2_batchnorm_readvariableop_resourceG
Cencoder_batch_normalization_2_batchnorm_mul_readvariableop_resourceE
Aencoder_batch_normalization_2_batchnorm_readvariableop_1_resourceE
Aencoder_batch_normalization_2_batchnorm_readvariableop_2_resource0
,encoder_dense_matmul_readvariableop_resource1
-encoder_dense_biasadd_readvariableop_resource
identityИЧ
$encoder/conv1d/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2&
$encoder/conv1d/conv1d/ExpandDims/dim┼
 encoder/conv1d/conv1d/ExpandDims
ExpandDimsinput_1-encoder/conv1d/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         А2"
 encoder/conv1d/conv1d/ExpandDimsх
1encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp:encoder_conv1d_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
:*
dtype023
1encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOpТ
&encoder/conv1d/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&encoder/conv1d/conv1d/ExpandDims_1/dimє
"encoder/conv1d/conv1d/ExpandDims_1
ExpandDims9encoder/conv1d/conv1d/ExpandDims_1/ReadVariableOp:value:0/encoder/conv1d/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
:2$
"encoder/conv1d/conv1d/ExpandDims_1Ї
encoder/conv1d/conv1dConv2D)encoder/conv1d/conv1d/ExpandDims:output:0+encoder/conv1d/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         №*
paddingVALID*
strides
2
encoder/conv1d/conv1d└
encoder/conv1d/conv1d/SqueezeSqueezeencoder/conv1d/conv1d:output:0*
T0*,
_output_shapes
:         №*
squeeze_dims

¤        2
encoder/conv1d/conv1d/Squeeze╣
%encoder/conv1d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv1d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%encoder/conv1d/BiasAdd/ReadVariableOp╔
encoder/conv1d/BiasAddBiasAdd&encoder/conv1d/conv1d/Squeeze:output:0-encoder/conv1d/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         №2
encoder/conv1d/BiasAddК
encoder/conv1d/TanhTanhencoder/conv1d/BiasAdd:output:0*
T0*,
_output_shapes
:         №2
encoder/conv1d/Tanhц
4encoder/batch_normalization/batchnorm/ReadVariableOpReadVariableOp=encoder_batch_normalization_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype026
4encoder/batch_normalization/batchnorm/ReadVariableOpЯ
+encoder/batch_normalization/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2-
+encoder/batch_normalization/batchnorm/add/y°
)encoder/batch_normalization/batchnorm/addAddV2<encoder/batch_normalization/batchnorm/ReadVariableOp:value:04encoder/batch_normalization/batchnorm/add/y:output:0*
T0*
_output_shapes
:2+
)encoder/batch_normalization/batchnorm/add╖
+encoder/batch_normalization/batchnorm/RsqrtRsqrt-encoder/batch_normalization/batchnorm/add:z:0*
T0*
_output_shapes
:2-
+encoder/batch_normalization/batchnorm/RsqrtЄ
8encoder/batch_normalization/batchnorm/mul/ReadVariableOpReadVariableOpAencoder_batch_normalization_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02:
8encoder/batch_normalization/batchnorm/mul/ReadVariableOpї
)encoder/batch_normalization/batchnorm/mulMul/encoder/batch_normalization/batchnorm/Rsqrt:y:0@encoder/batch_normalization/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2+
)encoder/batch_normalization/batchnorm/mulр
+encoder/batch_normalization/batchnorm/mul_1Mulencoder/conv1d/Tanh:y:0-encoder/batch_normalization/batchnorm/mul:z:0*
T0*,
_output_shapes
:         №2-
+encoder/batch_normalization/batchnorm/mul_1ь
6encoder/batch_normalization/batchnorm/ReadVariableOp_1ReadVariableOp?encoder_batch_normalization_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype028
6encoder/batch_normalization/batchnorm/ReadVariableOp_1ї
+encoder/batch_normalization/batchnorm/mul_2Mul>encoder/batch_normalization/batchnorm/ReadVariableOp_1:value:0-encoder/batch_normalization/batchnorm/mul:z:0*
T0*
_output_shapes
:2-
+encoder/batch_normalization/batchnorm/mul_2ь
6encoder/batch_normalization/batchnorm/ReadVariableOp_2ReadVariableOp?encoder_batch_normalization_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype028
6encoder/batch_normalization/batchnorm/ReadVariableOp_2є
)encoder/batch_normalization/batchnorm/subSub>encoder/batch_normalization/batchnorm/ReadVariableOp_2:value:0/encoder/batch_normalization/batchnorm/mul_2:z:0*
T0*
_output_shapes
:2+
)encoder/batch_normalization/batchnorm/sub·
+encoder/batch_normalization/batchnorm/add_1AddV2/encoder/batch_normalization/batchnorm/mul_1:z:0-encoder/batch_normalization/batchnorm/sub:z:0*
T0*,
_output_shapes
:         №2-
+encoder/batch_normalization/batchnorm/add_1Ы
&encoder/conv1d_1/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2(
&encoder/conv1d_1/conv1d/ExpandDims/dimє
"encoder/conv1d_1/conv1d/ExpandDims
ExpandDims/encoder/batch_normalization/batchnorm/add_1:z:0/encoder/conv1d_1/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         №2$
"encoder/conv1d_1/conv1d/ExpandDimsы
3encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<encoder_conv1d_1_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype025
3encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOpЦ
(encoder/conv1d_1/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(encoder/conv1d_1/conv1d/ExpandDims_1/dim√
$encoder/conv1d_1/conv1d/ExpandDims_1
ExpandDims;encoder/conv1d_1/conv1d/ExpandDims_1/ReadVariableOp:value:01encoder/conv1d_1/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2&
$encoder/conv1d_1/conv1d/ExpandDims_1№
encoder/conv1d_1/conv1dConv2D+encoder/conv1d_1/conv1d/ExpandDims:output:0-encoder/conv1d_1/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ° *
paddingVALID*
strides
2
encoder/conv1d_1/conv1d╞
encoder/conv1d_1/conv1d/SqueezeSqueeze encoder/conv1d_1/conv1d:output:0*
T0*,
_output_shapes
:         ° *
squeeze_dims

¤        2!
encoder/conv1d_1/conv1d/Squeeze┐
'encoder/conv1d_1/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv1d_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02)
'encoder/conv1d_1/BiasAdd/ReadVariableOp╤
encoder/conv1d_1/BiasAddBiasAdd(encoder/conv1d_1/conv1d/Squeeze:output:0/encoder/conv1d_1/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ° 2
encoder/conv1d_1/BiasAddР
encoder/conv1d_1/TanhTanh!encoder/conv1d_1/BiasAdd:output:0*
T0*,
_output_shapes
:         ° 2
encoder/conv1d_1/Tanhь
6encoder/batch_normalization_1/batchnorm/ReadVariableOpReadVariableOp?encoder_batch_normalization_1_batchnorm_readvariableop_resource*
_output_shapes
: *
dtype028
6encoder/batch_normalization_1/batchnorm/ReadVariableOpг
-encoder/batch_normalization_1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2/
-encoder/batch_normalization_1/batchnorm/add/yА
+encoder/batch_normalization_1/batchnorm/addAddV2>encoder/batch_normalization_1/batchnorm/ReadVariableOp:value:06encoder/batch_normalization_1/batchnorm/add/y:output:0*
T0*
_output_shapes
: 2-
+encoder/batch_normalization_1/batchnorm/add╜
-encoder/batch_normalization_1/batchnorm/RsqrtRsqrt/encoder/batch_normalization_1/batchnorm/add:z:0*
T0*
_output_shapes
: 2/
-encoder/batch_normalization_1/batchnorm/Rsqrt°
:encoder/batch_normalization_1/batchnorm/mul/ReadVariableOpReadVariableOpCencoder_batch_normalization_1_batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02<
:encoder/batch_normalization_1/batchnorm/mul/ReadVariableOp¤
+encoder/batch_normalization_1/batchnorm/mulMul1encoder/batch_normalization_1/batchnorm/Rsqrt:y:0Bencoder/batch_normalization_1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2-
+encoder/batch_normalization_1/batchnorm/mulш
-encoder/batch_normalization_1/batchnorm/mul_1Mulencoder/conv1d_1/Tanh:y:0/encoder/batch_normalization_1/batchnorm/mul:z:0*
T0*,
_output_shapes
:         ° 2/
-encoder/batch_normalization_1/batchnorm/mul_1Є
8encoder/batch_normalization_1/batchnorm/ReadVariableOp_1ReadVariableOpAencoder_batch_normalization_1_batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02:
8encoder/batch_normalization_1/batchnorm/ReadVariableOp_1¤
-encoder/batch_normalization_1/batchnorm/mul_2Mul@encoder/batch_normalization_1/batchnorm/ReadVariableOp_1:value:0/encoder/batch_normalization_1/batchnorm/mul:z:0*
T0*
_output_shapes
: 2/
-encoder/batch_normalization_1/batchnorm/mul_2Є
8encoder/batch_normalization_1/batchnorm/ReadVariableOp_2ReadVariableOpAencoder_batch_normalization_1_batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02:
8encoder/batch_normalization_1/batchnorm/ReadVariableOp_2√
+encoder/batch_normalization_1/batchnorm/subSub@encoder/batch_normalization_1/batchnorm/ReadVariableOp_2:value:01encoder/batch_normalization_1/batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2-
+encoder/batch_normalization_1/batchnorm/subВ
-encoder/batch_normalization_1/batchnorm/add_1AddV21encoder/batch_normalization_1/batchnorm/mul_1:z:0/encoder/batch_normalization_1/batchnorm/sub:z:0*
T0*,
_output_shapes
:         ° 2/
-encoder/batch_normalization_1/batchnorm/add_1Ы
&encoder/conv1d_2/conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2(
&encoder/conv1d_2/conv1d/ExpandDims/dimї
"encoder/conv1d_2/conv1d/ExpandDims
ExpandDims1encoder/batch_normalization_1/batchnorm/add_1:z:0/encoder/conv1d_2/conv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ° 2$
"encoder/conv1d_2/conv1d/ExpandDimsы
3encoder/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpReadVariableOp<encoder_conv1d_2_conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype025
3encoder/conv1d_2/conv1d/ExpandDims_1/ReadVariableOpЦ
(encoder/conv1d_2/conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2*
(encoder/conv1d_2/conv1d/ExpandDims_1/dim√
$encoder/conv1d_2/conv1d/ExpandDims_1
ExpandDims;encoder/conv1d_2/conv1d/ExpandDims_1/ReadVariableOp:value:01encoder/conv1d_2/conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2&
$encoder/conv1d_2/conv1d/ExpandDims_1№
encoder/conv1d_2/conv1dConv2D+encoder/conv1d_2/conv1d/ExpandDims:output:0-encoder/conv1d_2/conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         Ї@*
paddingVALID*
strides
2
encoder/conv1d_2/conv1d╞
encoder/conv1d_2/conv1d/SqueezeSqueeze encoder/conv1d_2/conv1d:output:0*
T0*,
_output_shapes
:         Ї@*
squeeze_dims

¤        2!
encoder/conv1d_2/conv1d/Squeeze┐
'encoder/conv1d_2/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv1d_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'encoder/conv1d_2/BiasAdd/ReadVariableOp╤
encoder/conv1d_2/BiasAddBiasAdd(encoder/conv1d_2/conv1d/Squeeze:output:0/encoder/conv1d_2/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         Ї@2
encoder/conv1d_2/BiasAddР
encoder/conv1d_2/TanhTanh!encoder/conv1d_2/BiasAdd:output:0*
T0*,
_output_shapes
:         Ї@2
encoder/conv1d_2/Tanhь
6encoder/batch_normalization_2/batchnorm/ReadVariableOpReadVariableOp?encoder_batch_normalization_2_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype028
6encoder/batch_normalization_2/batchnorm/ReadVariableOpг
-encoder/batch_normalization_2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2/
-encoder/batch_normalization_2/batchnorm/add/yА
+encoder/batch_normalization_2/batchnorm/addAddV2>encoder/batch_normalization_2/batchnorm/ReadVariableOp:value:06encoder/batch_normalization_2/batchnorm/add/y:output:0*
T0*
_output_shapes
:@2-
+encoder/batch_normalization_2/batchnorm/add╜
-encoder/batch_normalization_2/batchnorm/RsqrtRsqrt/encoder/batch_normalization_2/batchnorm/add:z:0*
T0*
_output_shapes
:@2/
-encoder/batch_normalization_2/batchnorm/Rsqrt°
:encoder/batch_normalization_2/batchnorm/mul/ReadVariableOpReadVariableOpCencoder_batch_normalization_2_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02<
:encoder/batch_normalization_2/batchnorm/mul/ReadVariableOp¤
+encoder/batch_normalization_2/batchnorm/mulMul1encoder/batch_normalization_2/batchnorm/Rsqrt:y:0Bencoder/batch_normalization_2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2-
+encoder/batch_normalization_2/batchnorm/mulш
-encoder/batch_normalization_2/batchnorm/mul_1Mulencoder/conv1d_2/Tanh:y:0/encoder/batch_normalization_2/batchnorm/mul:z:0*
T0*,
_output_shapes
:         Ї@2/
-encoder/batch_normalization_2/batchnorm/mul_1Є
8encoder/batch_normalization_2/batchnorm/ReadVariableOp_1ReadVariableOpAencoder_batch_normalization_2_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02:
8encoder/batch_normalization_2/batchnorm/ReadVariableOp_1¤
-encoder/batch_normalization_2/batchnorm/mul_2Mul@encoder/batch_normalization_2/batchnorm/ReadVariableOp_1:value:0/encoder/batch_normalization_2/batchnorm/mul:z:0*
T0*
_output_shapes
:@2/
-encoder/batch_normalization_2/batchnorm/mul_2Є
8encoder/batch_normalization_2/batchnorm/ReadVariableOp_2ReadVariableOpAencoder_batch_normalization_2_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02:
8encoder/batch_normalization_2/batchnorm/ReadVariableOp_2√
+encoder/batch_normalization_2/batchnorm/subSub@encoder/batch_normalization_2/batchnorm/ReadVariableOp_2:value:01encoder/batch_normalization_2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2-
+encoder/batch_normalization_2/batchnorm/subВ
-encoder/batch_normalization_2/batchnorm/add_1AddV21encoder/batch_normalization_2/batchnorm/mul_1:z:0/encoder/batch_normalization_2/batchnorm/sub:z:0*
T0*,
_output_shapes
:         Ї@2/
-encoder/batch_normalization_2/batchnorm/add_1
encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"     }  2
encoder/flatten/Const─
encoder/flatten/ReshapeReshape1encoder/batch_normalization_2/batchnorm/add_1:z:0encoder/flatten/Const:output:0*
T0*)
_output_shapes
:         А·2
encoder/flatten/Reshape╣
#encoder/dense/MatMul/ReadVariableOpReadVariableOp,encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
А·*
dtype02%
#encoder/dense/MatMul/ReadVariableOp╖
encoder/dense/MatMulMatMul encoder/flatten/Reshape:output:0+encoder/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
encoder/dense/MatMul╢
$encoder/dense/BiasAdd/ReadVariableOpReadVariableOp-encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$encoder/dense/BiasAdd/ReadVariableOp╣
encoder/dense/BiasAddBiasAddencoder/dense/MatMul:product:0,encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
encoder/dense/BiasAddr
IdentityIdentityencoder/dense/BiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*{
_input_shapesj
h:         А:::::::::::::::::::::U Q
,
_output_shapes
:         А
!
_user_specified_name	input_1
ч
и
5__inference_batch_normalization_layer_call_fn_9201546

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_91996802
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :                  2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                  ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
═
к
7__inference_batch_normalization_2_layer_call_fn_9201887

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallб
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_92003412
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         Ї@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         Ї@::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
щ)
═
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9200198

inputs
assignmovingavg_9200173
assignmovingavg_1_9200179)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         ° 2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1Я
AssignMovingAvg/decayConst**
_class 
loc:@AssignMovingAvg/9200173*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9200173*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp─
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0**
_class 
loc:@AssignMovingAvg/9200173*
_output_shapes
: 2
AssignMovingAvg/sub╗
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg/9200173*
_output_shapes
: 2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9200173AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/9200173*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/decayConst*,
_class"
 loc:@AssignMovingAvg_1/9200179*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9200179*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp╬
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9200179*
_output_shapes
: 2
AssignMovingAvg_1/sub┼
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9200179*
_output_shapes
: 2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9200179AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/9200179*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         ° 2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         ° 2
batchnorm/add_1║
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*,
_output_shapes
:         ° 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         ° ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:T P
,
_output_shapes
:         ° 
 
_user_specified_nameinputs
а
E
)__inference_flatten_layer_call_fn_9201383

inputs
identity─
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         А·* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_92003832
PartitionedCalln
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:         А·2

Identity"
identityIdentity:output:0*+
_input_shapes
:         Ї@:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
Щ
║
E__inference_conv1d_2_layer_call_and_return_conditional_losses_9200270

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ° 2
conv1d/ExpandDims╕
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d/ExpandDims_1╕
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         Ї@*
paddingVALID*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         Ї@*
squeeze_dims

¤        2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         Ї@2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         Ї@2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:         Ї@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ° :::T P
,
_output_shapes
:         ° 
 
_user_specified_nameinputs
╦
к
7__inference_batch_normalization_2_layer_call_fn_9201874

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_92003212
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         Ї@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         Ї@::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
╕
`
D__inference_flatten_layer_call_and_return_conditional_losses_9201378

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"     }  2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:         А·2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:         А·2

Identity"
identityIdentity:output:0*+
_input_shapes
:         Ї@:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
╤
к
B__inference_dense_layer_call_and_return_conditional_losses_9201393

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИП
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
А·*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*0
_input_shapes
:         А·:::Q M
)
_output_shapes
:         А·
 
_user_specified_nameinputs
Э
Х
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9200341

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         Ї@2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         Ї@2
batchnorm/add_1l
IdentityIdentitybatchnorm/add_1:z:0*
T0*,
_output_shapes
:         Ї@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         Ї@:::::T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
╦
к
7__inference_batch_normalization_1_layer_call_fn_9201710

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЯ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ° *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_92001982
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         ° 2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         ° ::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         ° 
 
_user_specified_nameinputs
ю
}
(__inference_conv1d_layer_call_fn_9201427

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         №*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_layer_call_and_return_conditional_losses_92000242
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         №2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         А::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
Щ
║
E__inference_conv1d_1_layer_call_and_return_conditional_losses_9200147

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         №2
conv1d/ExpandDims╕
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: *
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: 2
conv1d/ExpandDims_1╕
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         ° *
paddingVALID*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         ° *
squeeze_dims

¤        2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         ° 2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         ° 2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:         ° 2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         №:::T P
,
_output_shapes
:         №
 
_user_specified_nameinputs
Ъ*
═
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9201759

inputs
assignmovingavg_9201734
assignmovingavg_1_9201740)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
: 2
moments/StopGradient▒
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                   2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
: *
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
: *
squeeze_dims
 2
moments/Squeeze_1Я
AssignMovingAvg/decayConst**
_class 
loc:@AssignMovingAvg/9201734*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9201734*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp─
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0**
_class 
loc:@AssignMovingAvg/9201734*
_output_shapes
: 2
AssignMovingAvg/sub╗
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg/9201734*
_output_shapes
: 2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9201734AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/9201734*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/decayConst*,
_class"
 loc:@AssignMovingAvg_1/9201740*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9201740*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp╬
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9201740*
_output_shapes
: 2
AssignMovingAvg_1/sub┼
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9201740*
_output_shapes
: 2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9201740AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/9201740*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                   2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                   2
batchnorm/add_1┬
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*4
_output_shapes"
 :                   2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                   ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
╟
и
5__inference_batch_normalization_layer_call_fn_9201628

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         №*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_92000752
StatefulPartitionedCallУ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:         №2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         №::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:         №
 
_user_specified_nameinputs
Б/
║
D__inference_encoder_layer_call_and_return_conditional_losses_9200525

inputs
conv1d_9200476
conv1d_9200478
batch_normalization_9200481
batch_normalization_9200483
batch_normalization_9200485
batch_normalization_9200487
conv1d_1_9200490
conv1d_1_9200492!
batch_normalization_1_9200495!
batch_normalization_1_9200497!
batch_normalization_1_9200499!
batch_normalization_1_9200501
conv1d_2_9200504
conv1d_2_9200506!
batch_normalization_2_9200509!
batch_normalization_2_9200511!
batch_normalization_2_9200513!
batch_normalization_2_9200515
dense_9200519
dense_9200521
identityИв+batch_normalization/StatefulPartitionedCallв-batch_normalization_1/StatefulPartitionedCallв-batch_normalization_2/StatefulPartitionedCallвconv1d/StatefulPartitionedCallв conv1d_1/StatefulPartitionedCallв conv1d_2/StatefulPartitionedCallвdense/StatefulPartitionedCallТ
conv1d/StatefulPartitionedCallStatefulPartitionedCallinputsconv1d_9200476conv1d_9200478*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         №*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv1d_layer_call_and_return_conditional_losses_92000242 
conv1d/StatefulPartitionedCall░
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall'conv1d/StatefulPartitionedCall:output:0batch_normalization_9200481batch_normalization_9200483batch_normalization_9200485batch_normalization_9200487*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         №*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Y
fTRR
P__inference_batch_normalization_layer_call_and_return_conditional_losses_92000752-
+batch_normalization/StatefulPartitionedCall╩
 conv1d_1/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv1d_1_9200490conv1d_1_9200492*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ° *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv1d_1_layer_call_and_return_conditional_losses_92001472"
 conv1d_1/StatefulPartitionedCall└
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall)conv1d_1/StatefulPartitionedCall:output:0batch_normalization_1_9200495batch_normalization_1_9200497batch_normalization_1_9200499batch_normalization_1_9200501*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         ° *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_92001982/
-batch_normalization_1/StatefulPartitionedCall╠
 conv1d_2/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0conv1d_2_9200504conv1d_2_9200506*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *N
fIRG
E__inference_conv1d_2_layer_call_and_return_conditional_losses_92002702"
 conv1d_2/StatefulPartitionedCall└
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall)conv1d_2/StatefulPartitionedCall:output:0batch_normalization_2_9200509batch_normalization_2_9200511batch_normalization_2_9200513batch_normalization_2_9200515*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:         Ї@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_92003212/
-batch_normalization_2/StatefulPartitionedCallД
flatten/PartitionedCallPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:         А·* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_92003832
flatten/PartitionedCallв
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_9200519dense_9200521*
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
GPU 2J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_92004012
dense/StatefulPartitionedCallП
IdentityIdentity&dense/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall^conv1d/StatefulPartitionedCall!^conv1d_1/StatefulPartitionedCall!^conv1d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*{
_input_shapesj
h:         А::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2@
conv1d/StatefulPartitionedCallconv1d/StatefulPartitionedCall2D
 conv1d_1/StatefulPartitionedCall conv1d_1/StatefulPartitionedCall2D
 conv1d_2/StatefulPartitionedCall conv1d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:T P
,
_output_shapes
:         А
 
_user_specified_nameinputs
э
к
7__inference_batch_normalization_1_layer_call_fn_9201805

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                   *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_91998532
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :                   2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                   ::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
Ъ*
═
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9199960

inputs
assignmovingavg_9199935
assignmovingavg_1_9199941)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradient▒
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*4
_output_shapes"
 :                  @2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1Я
AssignMovingAvg/decayConst**
_class 
loc:@AssignMovingAvg/9199935*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9199935*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp─
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0**
_class 
loc:@AssignMovingAvg/9199935*
_output_shapes
:@2
AssignMovingAvg/sub╗
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg/9199935*
_output_shapes
:@2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9199935AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/9199935*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/decayConst*,
_class"
 loc:@AssignMovingAvg_1/9199941*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9199941*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╬
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9199941*
_output_shapes
:@2
AssignMovingAvg_1/sub┼
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9199941*
_output_shapes
:@2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9199941AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/9199941*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                  @2
batchnorm/add_1┬
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*4
_output_shapes"
 :                  @2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                  @::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
ч)
╦
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9201595

inputs
assignmovingavg_9201570
assignmovingavg_1_9201576)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:2
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         №2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1Я
AssignMovingAvg/decayConst**
_class 
loc:@AssignMovingAvg/9201570*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9201570*
_output_shapes
:*
dtype02 
AssignMovingAvg/ReadVariableOp─
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0**
_class 
loc:@AssignMovingAvg/9201570*
_output_shapes
:2
AssignMovingAvg/sub╗
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg/9201570*
_output_shapes
:2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9201570AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/9201570*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/decayConst*,
_class"
 loc:@AssignMovingAvg_1/9201576*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9201576*
_output_shapes
:*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╬
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9201576*
_output_shapes
:2
AssignMovingAvg_1/sub┼
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9201576*
_output_shapes
:2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9201576AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/9201576*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         №2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         №2
batchnorm/add_1║
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*,
_output_shapes
:         №2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         №::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:T P
,
_output_shapes
:         №
 
_user_specified_nameinputs
Щ
║
E__inference_conv1d_2_layer_call_and_return_conditional_losses_9201468

inputs/
+conv1d_expanddims_1_readvariableop_resource#
biasadd_readvariableop_resource
identityИy
conv1d/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
¤        2
conv1d/ExpandDims/dimЧ
conv1d/ExpandDims
ExpandDimsinputsconv1d/ExpandDims/dim:output:0*
T0*0
_output_shapes
:         ° 2
conv1d/ExpandDims╕
"conv1d/ExpandDims_1/ReadVariableOpReadVariableOp+conv1d_expanddims_1_readvariableop_resource*"
_output_shapes
: @*
dtype02$
"conv1d/ExpandDims_1/ReadVariableOpt
conv1d/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B : 2
conv1d/ExpandDims_1/dim╖
conv1d/ExpandDims_1
ExpandDims*conv1d/ExpandDims_1/ReadVariableOp:value:0 conv1d/ExpandDims_1/dim:output:0*
T0*&
_output_shapes
: @2
conv1d/ExpandDims_1╕
conv1dConv2Dconv1d/ExpandDims:output:0conv1d/ExpandDims_1:output:0*
T0*0
_output_shapes
:         Ї@*
paddingVALID*
strides
2
conv1dУ
conv1d/SqueezeSqueezeconv1d:output:0*
T0*,
_output_shapes
:         Ї@*
squeeze_dims

¤        2
conv1d/SqueezeМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpН
BiasAddBiasAddconv1d/Squeeze:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:         Ї@2	
BiasAdd]
TanhTanhBiasAdd:output:0*
T0*,
_output_shapes
:         Ї@2
Tanha
IdentityIdentityTanh:y:0*
T0*,
_output_shapes
:         Ї@2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :         ° :::T P
,
_output_shapes
:         ° 
 
_user_specified_nameinputs
щ)
═
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9201841

inputs
assignmovingavg_9201816
assignmovingavg_1_9201822)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         Ї@2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1Я
AssignMovingAvg/decayConst**
_class 
loc:@AssignMovingAvg/9201816*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9201816*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp─
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0**
_class 
loc:@AssignMovingAvg/9201816*
_output_shapes
:@2
AssignMovingAvg/sub╗
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg/9201816*
_output_shapes
:@2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9201816AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/9201816*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/decayConst*,
_class"
 loc:@AssignMovingAvg_1/9201822*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9201822*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╬
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9201822*
_output_shapes
:@2
AssignMovingAvg_1/sub┼
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9201822*
_output_shapes
:@2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9201822AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/9201822*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         Ї@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         Ї@2
batchnorm/add_1║
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*,
_output_shapes
:         Ї@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         Ї@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs
э
к
7__inference_batch_normalization_2_layer_call_fn_9201969

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityИвStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *[
fVRT
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_91999932
StatefulPartitionedCallЫ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :                  @2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                  @::::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
╞
Х
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9201779

inputs%
!batchnorm_readvariableop_resource)
%batchnorm_mul_readvariableop_resource'
#batchnorm_readvariableop_1_resource'
#batchnorm_readvariableop_2_resource
identityИТ
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yИ
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
: 2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
: 2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
: *
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
: 2
batchnorm/mulГ
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*4
_output_shapes"
 :                   2
batchnorm/mul_1Ш
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_1Е
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
: 2
batchnorm/mul_2Ш
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
: *
dtype02
batchnorm/ReadVariableOp_2Г
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
: 2
batchnorm/subТ
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*4
_output_shapes"
 :                   2
batchnorm/add_1t
IdentityIdentitybatchnorm/add_1:z:0*
T0*4
_output_shapes"
 :                   2

Identity"
identityIdentity:output:0*C
_input_shapes2
0:                   :::::\ X
4
_output_shapes"
 :                   
 
_user_specified_nameinputs
щ)
═
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9200321

inputs
assignmovingavg_9200296
assignmovingavg_1_9200302)
%batchnorm_mul_readvariableop_resource%
!batchnorm_readvariableop_resource
identityИв#AssignMovingAvg/AssignSubVariableOpв%AssignMovingAvg_1/AssignSubVariableOpС
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2 
moments/mean/reduction_indicesУ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/meanА
moments/StopGradientStopGradientmoments/mean:output:0*
T0*"
_output_shapes
:@2
moments/StopGradientй
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*,
_output_shapes
:         Ї@2
moments/SquaredDifferenceЩ
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB"       2$
"moments/variance/reduction_indices╢
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*"
_output_shapes
:@*
	keep_dims(2
moments/varianceБ
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/SqueezeЙ
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 2
moments/Squeeze_1Я
AssignMovingAvg/decayConst**
_class 
loc:@AssignMovingAvg/9200296*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg/decayФ
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_9200296*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp─
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0**
_class 
loc:@AssignMovingAvg/9200296*
_output_shapes
:@2
AssignMovingAvg/sub╗
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0**
_class 
loc:@AssignMovingAvg/9200296*
_output_shapes
:@2
AssignMovingAvg/mulГ
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_9200296AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp**
_class 
loc:@AssignMovingAvg/9200296*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOpе
AssignMovingAvg_1/decayConst*,
_class"
 loc:@AssignMovingAvg_1/9200302*
_output_shapes
: *
dtype0*
valueB
 *
╫#<2
AssignMovingAvg_1/decayЪ
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_9200302*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp╬
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9200302*
_output_shapes
:@2
AssignMovingAvg_1/sub┼
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*,
_class"
 loc:@AssignMovingAvg_1/9200302*
_output_shapes
:@2
AssignMovingAvg_1/mulП
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_9200302AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*,
_class"
 loc:@AssignMovingAvg_1/9200302*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOpg
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:2
batchnorm/add/yВ
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:@2
batchnorm/addc
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@2
batchnorm/RsqrtЮ
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/mul/ReadVariableOpЕ
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@2
batchnorm/mul{
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*,
_output_shapes
:         Ї@2
batchnorm/mul_1{
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@2
batchnorm/mul_2Т
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype02
batchnorm/ReadVariableOpБ
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@2
batchnorm/subК
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*,
_output_shapes
:         Ї@2
batchnorm/add_1║
IdentityIdentitybatchnorm/add_1:z:0$^AssignMovingAvg/AssignSubVariableOp&^AssignMovingAvg_1/AssignSubVariableOp*
T0*,
_output_shapes
:         Ї@2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:         Ї@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp:T P
,
_output_shapes
:         Ї@
 
_user_specified_nameinputs"╕L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*░
serving_defaultЬ
@
input_15
serving_default_input_1:0         А<
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:оя
▐
	convs
	norms
flatten
out
regularization_losses
trainable_variables
	variables
	keras_api
	
signatures
t__call__
*u&call_and_return_all_conditional_losses
v_default_save_signature"Ў
_tf_keras_model▄{"class_name": "Encoder", "name": "encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Encoder"}}
5

0
1
2"
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
т
regularization_losses
trainable_variables
	variables
	keras_api
w__call__
*x&call_and_return_all_conditional_losses"╙
_tf_keras_layer╣{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
Є

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
y__call__
*z&call_and_return_all_conditional_losses"═
_tf_keras_layer│{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32000}}}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 32000]}}
 "
trackable_list_wrapper
Ж
0
1
2
3
4
5
 6
!7
"8
#9
$10
%11
12
13"
trackable_list_wrapper
╢
0
1
2
3
4
5
 6
!7
"8
#9
$10
%11
&12
'13
(14
)15
*16
+17
18
19"
trackable_list_wrapper
╩
,metrics
regularization_losses
-layer_metrics
.layer_regularization_losses
trainable_variables
/non_trainable_variables

0layers
	variables
t__call__
v_default_save_signature
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
,
{serving_default"
signature_map
▀	

kernel
bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
|__call__
*}&call_and_return_all_conditional_losses"║
_tf_keras_layerа{"class_name": "Conv1D", "name": "conv1d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 512, 1]}}
х	

kernel
bias
5regularization_losses
6trainable_variables
7	variables
8	keras_api
~__call__
*&call_and_return_all_conditional_losses"└
_tf_keras_layerж{"class_name": "Conv1D", "name": "conv1d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 508, 16]}}
ч	

kernel
bias
9regularization_losses
:trainable_variables
;	variables
<	keras_api
А__call__
+Б&call_and_return_all_conditional_losses"└
_tf_keras_layerж{"class_name": "Conv1D", "name": "conv1d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv1d_2", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [5]}, "strides": {"class_name": "__tuple__", "items": [1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 3, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 504, 32]}}
│	
=axis
	 gamma
!beta
&moving_mean
'moving_variance
>regularization_losses
?trainable_variables
@	variables
A	keras_api
В__call__
+Г&call_and_return_all_conditional_losses"▌
_tf_keras_layer├{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 508, 16]}}
╖	
Baxis
	"gamma
#beta
(moving_mean
)moving_variance
Cregularization_losses
Dtrainable_variables
E	variables
F	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses"с
_tf_keras_layer╟{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 504, 32]}}
╖	
Gaxis
	$gamma
%beta
*moving_mean
+moving_variance
Hregularization_losses
Itrainable_variables
J	variables
K	keras_api
Ж__call__
+З&call_and_return_all_conditional_losses"с
_tf_keras_layer╟{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [2], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {"2": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [50, 500, 64]}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
Lmetrics
Mlayer_metrics
regularization_losses
Nlayer_regularization_losses
trainable_variables
Onon_trainable_variables

Players
	variables
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
(:&
А·2encoder/dense/kernel
 :2encoder/dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
н
Qmetrics
Rlayer_metrics
regularization_losses
Slayer_regularization_losses
trainable_variables
Tnon_trainable_variables

Ulayers
	variables
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
+:)2encoder/conv1d/kernel
!:2encoder/conv1d/bias
-:+ 2encoder/conv1d_1/kernel
#:! 2encoder/conv1d_1/bias
-:+ @2encoder/conv1d_2/kernel
#:!@2encoder/conv1d_2/bias
/:-2!encoder/batch_normalization/gamma
.:,2 encoder/batch_normalization/beta
1:/ 2#encoder/batch_normalization_1/gamma
0:. 2"encoder/batch_normalization_1/beta
1:/@2#encoder/batch_normalization_2/gamma
0:.@2"encoder/batch_normalization_2/beta
7:5 (2'encoder/batch_normalization/moving_mean
;:9 (2+encoder/batch_normalization/moving_variance
9:7  (2)encoder/batch_normalization_1/moving_mean
=:;  (2-encoder/batch_normalization_1/moving_variance
9:7@ (2)encoder/batch_normalization_2/moving_mean
=:;@ (2-encoder/batch_normalization_2/moving_variance
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
J
&0
'1
(2
)3
*4
+5"
trackable_list_wrapper
X

0
1
2
3
4
5
6
7"
trackable_list_wrapper
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
н
Vmetrics
Wlayer_metrics
1regularization_losses
Xlayer_regularization_losses
2trainable_variables
Ynon_trainable_variables

Zlayers
3	variables
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
н
[metrics
\layer_metrics
5regularization_losses
]layer_regularization_losses
6trainable_variables
^non_trainable_variables

_layers
7	variables
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
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
`metrics
alayer_metrics
9regularization_losses
blayer_regularization_losses
:trainable_variables
cnon_trainable_variables

dlayers
;	variables
А__call__
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
<
 0
!1
&2
'3"
trackable_list_wrapper
░
emetrics
flayer_metrics
>regularization_losses
glayer_regularization_losses
?trainable_variables
hnon_trainable_variables

ilayers
@	variables
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
<
"0
#1
(2
)3"
trackable_list_wrapper
░
jmetrics
klayer_metrics
Cregularization_losses
llayer_regularization_losses
Dtrainable_variables
mnon_trainable_variables

nlayers
E	variables
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
<
$0
%1
*2
+3"
trackable_list_wrapper
░
ometrics
player_metrics
Hregularization_losses
qlayer_regularization_losses
Itrainable_variables
rnon_trainable_variables

slayers
J	variables
Ж__call__
+З&call_and_return_all_conditional_losses
'З"call_and_return_conditional_losses"
_generic_user_object
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
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
ц2у
)__inference_encoder_layer_call_fn_9201327
)__inference_encoder_layer_call_fn_9201372
)__inference_encoder_layer_call_fn_9200997
)__inference_encoder_layer_call_fn_9201042┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╥2╧
D__inference_encoder_layer_call_and_return_conditional_losses_9201282
D__inference_encoder_layer_call_and_return_conditional_losses_9201186
D__inference_encoder_layer_call_and_return_conditional_losses_9200952
D__inference_encoder_layer_call_and_return_conditional_losses_9200856┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
х2т
"__inference__wrapped_model_9199584╗
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *+в(
&К#
input_1         А
╙2╨
)__inference_flatten_layer_call_fn_9201383в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ю2ы
D__inference_flatten_layer_call_and_return_conditional_losses_9201378в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╤2╬
'__inference_dense_layer_call_fn_9201402в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ь2щ
B__inference_dense_layer_call_and_return_conditional_losses_9201393в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
4B2
%__inference_signature_wrapper_9200712input_1
╥2╧
(__inference_conv1d_layer_call_fn_9201427в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_conv1d_layer_call_and_return_conditional_losses_9201418в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_conv1d_1_layer_call_fn_9201452в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_conv1d_1_layer_call_and_return_conditional_losses_9201443в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╘2╤
*__inference_conv1d_2_layer_call_fn_9201477в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
я2ь
E__inference_conv1d_2_layer_call_and_return_conditional_losses_9201468в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ц2У
5__inference_batch_normalization_layer_call_fn_9201628
5__inference_batch_normalization_layer_call_fn_9201559
5__inference_batch_normalization_layer_call_fn_9201641
5__inference_batch_normalization_layer_call_fn_9201546┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
В2 
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9201595
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9201513
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9201615
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9201533┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ю2Ы
7__inference_batch_normalization_1_layer_call_fn_9201723
7__inference_batch_normalization_1_layer_call_fn_9201710
7__inference_batch_normalization_1_layer_call_fn_9201792
7__inference_batch_normalization_1_layer_call_fn_9201805┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
К2З
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9201779
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9201759
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9201697
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9201677┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
Ю2Ы
7__inference_batch_normalization_2_layer_call_fn_9201874
7__inference_batch_normalization_2_layer_call_fn_9201969
7__inference_batch_normalization_2_layer_call_fn_9201956
7__inference_batch_normalization_2_layer_call_fn_9201887┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
К2З
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9201923
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9201861
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9201841
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9201943┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 й
"__inference__wrapped_model_9199584В' &!)"(#+$*%5в2
+в(
&К#
input_1         А
к "3к0
.
output_1"К
output_1         ┬
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9201677l()"#8в5
.в+
%К"
inputs         ° 
p
к "*в'
 К
0         ° 
Ъ ┬
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9201697l)"(#8в5
.в+
%К"
inputs         ° 
p 
к "*в'
 К
0         ° 
Ъ ╥
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9201759|()"#@в=
6в3
-К*
inputs                   
p
к "2в/
(К%
0                   
Ъ ╥
R__inference_batch_normalization_1_layer_call_and_return_conditional_losses_9201779|)"(#@в=
6в3
-К*
inputs                   
p 
к "2в/
(К%
0                   
Ъ Ъ
7__inference_batch_normalization_1_layer_call_fn_9201710_()"#8в5
.в+
%К"
inputs         ° 
p
к "К         ° Ъ
7__inference_batch_normalization_1_layer_call_fn_9201723_)"(#8в5
.в+
%К"
inputs         ° 
p 
к "К         ° к
7__inference_batch_normalization_1_layer_call_fn_9201792o()"#@в=
6в3
-К*
inputs                   
p
к "%К"                   к
7__inference_batch_normalization_1_layer_call_fn_9201805o)"(#@в=
6в3
-К*
inputs                   
p 
к "%К"                   ┬
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9201841l*+$%8в5
.в+
%К"
inputs         Ї@
p
к "*в'
 К
0         Ї@
Ъ ┬
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9201861l+$*%8в5
.в+
%К"
inputs         Ї@
p 
к "*в'
 К
0         Ї@
Ъ ╥
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9201923|*+$%@в=
6в3
-К*
inputs                  @
p
к "2в/
(К%
0                  @
Ъ ╥
R__inference_batch_normalization_2_layer_call_and_return_conditional_losses_9201943|+$*%@в=
6в3
-К*
inputs                  @
p 
к "2в/
(К%
0                  @
Ъ Ъ
7__inference_batch_normalization_2_layer_call_fn_9201874_*+$%8в5
.в+
%К"
inputs         Ї@
p
к "К         Ї@Ъ
7__inference_batch_normalization_2_layer_call_fn_9201887_+$*%8в5
.в+
%К"
inputs         Ї@
p 
к "К         Ї@к
7__inference_batch_normalization_2_layer_call_fn_9201956o*+$%@в=
6в3
-К*
inputs                  @
p
к "%К"                  @к
7__inference_batch_normalization_2_layer_call_fn_9201969o+$*%@в=
6в3
-К*
inputs                  @
p 
к "%К"                  @╨
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9201513|&' !@в=
6в3
-К*
inputs                  
p
к "2в/
(К%
0                  
Ъ ╨
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9201533|' &!@в=
6в3
-К*
inputs                  
p 
к "2в/
(К%
0                  
Ъ └
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9201595l&' !8в5
.в+
%К"
inputs         №
p
к "*в'
 К
0         №
Ъ └
P__inference_batch_normalization_layer_call_and_return_conditional_losses_9201615l' &!8в5
.в+
%К"
inputs         №
p 
к "*в'
 К
0         №
Ъ и
5__inference_batch_normalization_layer_call_fn_9201546o&' !@в=
6в3
-К*
inputs                  
p
к "%К"                  и
5__inference_batch_normalization_layer_call_fn_9201559o' &!@в=
6в3
-К*
inputs                  
p 
к "%К"                  Ш
5__inference_batch_normalization_layer_call_fn_9201628_&' !8в5
.в+
%К"
inputs         №
p
к "К         №Ш
5__inference_batch_normalization_layer_call_fn_9201641_' &!8в5
.в+
%К"
inputs         №
p 
к "К         №п
E__inference_conv1d_1_layer_call_and_return_conditional_losses_9201443f4в1
*в'
%К"
inputs         №
к "*в'
 К
0         ° 
Ъ З
*__inference_conv1d_1_layer_call_fn_9201452Y4в1
*в'
%К"
inputs         №
к "К         ° п
E__inference_conv1d_2_layer_call_and_return_conditional_losses_9201468f4в1
*в'
%К"
inputs         ° 
к "*в'
 К
0         Ї@
Ъ З
*__inference_conv1d_2_layer_call_fn_9201477Y4в1
*в'
%К"
inputs         ° 
к "К         Ї@н
C__inference_conv1d_layer_call_and_return_conditional_losses_9201418f4в1
*в'
%К"
inputs         А
к "*в'
 К
0         №
Ъ Е
(__inference_conv1d_layer_call_fn_9201427Y4в1
*в'
%К"
inputs         А
к "К         №д
B__inference_dense_layer_call_and_return_conditional_losses_9201393^1в.
'в$
"К
inputs         А·
к "%в"
К
0         
Ъ |
'__inference_dense_layer_call_fn_9201402Q1в.
'в$
"К
inputs         А·
к "К         └
D__inference_encoder_layer_call_and_return_conditional_losses_9200856x&' !()"#*+$%9в6
/в,
&К#
input_1         А
p
к "%в"
К
0         
Ъ └
D__inference_encoder_layer_call_and_return_conditional_losses_9200952x' &!)"(#+$*%9в6
/в,
&К#
input_1         А
p 
к "%в"
К
0         
Ъ ┐
D__inference_encoder_layer_call_and_return_conditional_losses_9201186w&' !()"#*+$%8в5
.в+
%К"
inputs         А
p
к "%в"
К
0         
Ъ ┐
D__inference_encoder_layer_call_and_return_conditional_losses_9201282w' &!)"(#+$*%8в5
.в+
%К"
inputs         А
p 
к "%в"
К
0         
Ъ Ш
)__inference_encoder_layer_call_fn_9200997k&' !()"#*+$%9в6
/в,
&К#
input_1         А
p
к "К         Ш
)__inference_encoder_layer_call_fn_9201042k' &!)"(#+$*%9в6
/в,
&К#
input_1         А
p 
к "К         Ч
)__inference_encoder_layer_call_fn_9201327j&' !()"#*+$%8в5
.в+
%К"
inputs         А
p
к "К         Ч
)__inference_encoder_layer_call_fn_9201372j' &!)"(#+$*%8в5
.в+
%К"
inputs         А
p 
к "К         з
D__inference_flatten_layer_call_and_return_conditional_losses_9201378_4в1
*в'
%К"
inputs         Ї@
к "'в$
К
0         А·
Ъ 
)__inference_flatten_layer_call_fn_9201383R4в1
*в'
%К"
inputs         Ї@
к "К         А·╖
%__inference_signature_wrapper_9200712Н' &!)"(#+$*%@в=
в 
6к3
1
input_1&К#
input_1         А"3к0
.
output_1"К
output_1         