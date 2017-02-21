import copy, numpy as np
np.random.seed(0)
 
# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output
 
# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output*(1-output)
 

# training dataset generation
int2binary = {}
binary_dim = 8
 
largest_number = pow(2,binary_dim)
binary = np.unpackbits(
    np.array([range(largest_number)],dtype=np.uint8).T,axis=1)
for i in range(largest_number):
    int2binary[i] = binary[i]
 

# input variables
alpha = 0.1
input_dim = 2
hidden_dim = 16
output_dim = 1
 

# initialize neural network weights
synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1
 
synapse_0_update = np.zeros_like(synapse_0)
synapse_1_update = np.zeros_like(synapse_1)
synapse_h_update = np.zeros_like(synapse_h)
 
# training logic
for j in range(10000):
 
    # generate a simple addition problem (a + b = c)
    a_int = np.random.randint(largest_number/2) # int version
    a = int2binary[a_int] # binary encoding
 
    b_int = np.random.randint(largest_number/2) # int version
    b = int2binary[b_int] # binary encoding
 
    # true answer
    c_int = a_int + b_int
    c = int2binary[c_int]
 
    # where we'll store our best guess (binary encoded)
    d = np.zeros_like(c)
 
    overallError = 0
 
    layer_2_deltas = list()
    layer_1_values = list()
    layer_1_values.append(np.zeros(hidden_dim))
 
    # moving along the positions in the binary encoding
    for position in range(binary_dim):
 
        # generate input and output
        X = np.array([[a[binary_dim - position - 1],b[binary_dim - position - 1]]])
        y = np.array([[c[binary_dim - position - 1]]]).T
 
        # hidden layer (input ~+ prev_hidden)
        layer_1 = sigmoid(np.dot(X,synapse_0) + np.dot(layer_1_values[-1],synapse_h))
 
        # output layer (new binary representation)
        layer_2 = sigmoid(np.dot(layer_1,synapse_1))
 
        # did we miss?... if so by how much?
        layer_2_error = y - layer_2
        layer_2_deltas.append((layer_2_error)*sigmoid_output_to_derivative(layer_2))
        overallError += np.abs(layer_2_error[0])
 
        # decode estimate so we can print it out
        d[binary_dim - position - 1] = np.round(layer_2[0][0])
 
        # store hidden layer so we can use it in the next timestep
        layer_1_values.append(copy.deepcopy(layer_1))
 
    future_layer_1_delta = np.zeros(hidden_dim)
 
    for position in range(binary_dim):
 
        X = np.array([[a[position],b[position]]])
        layer_1 = layer_1_values[-position-1]
        prev_layer_1 = layer_1_values[-position-2]
 
        # error at output layer
        layer_2_delta = layer_2_deltas[-position-1]
        # error at hidden layer
        layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + 
            layer_2_delta.dot(synapse_1.T)) * sigmoid_output_to_derivative(layer_1)
        # let's update all our weights so we can try again
        synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
        synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)
        synapse_0_update += X.T.dot(layer_1_delta)
 
        future_layer_1_delta = layer_1_delta
 
    
    synapse_0 += synapse_0_update * alpha
    synapse_1 += synapse_1_update * alpha
    synapse_h += synapse_h_update * alpha    
 
    synapse_0_update *= 0
    synapse_1_update *= 0
    synapse_h_update *= 0
 

    # print out progress
    if(j % 1000 == 0):
        print "Error:" + str(overallError)
        print "Pred:" + str(d)
        print "True:" + str(c)
        out = 0
        for index,x in enumerate(reversed(d)):
            out += x*pow(2,index)
        print str(a_int) + " + " + str(b_int) + " = " + str(out)
        print "------------"



'''
Lines 0-2:导入依赖包，设定随机数生成的种子。我们只需要两个依赖包，numpy和copy。numpy是为了矩阵计算，copy用来拷贝东西。

Lines 4-11:我们的非线性函数与其导数，更多的细节可见参考我们之前的博客：http://blog.csdn.net/zzukun/article/details/49556715

Line 15:这一行声明了一个查找表，这个表是一个实数与对应二进制表示的映射。二进制表示将会是我们网路的输入与输出，所以这个查找表将会帮助我们将实数转化为其二进制表示。

Line 16:这里设置了二进制数的最大长度。如果一切都调试好了，你可以把它调整为一个非常大的数。

Line 18:这里计算了跟二进制最大长度对应的可以表示的最大十进制数。

Line 19:这里生成了十进制数转二进制数的查找表，并将其复制到int2binary里面。虽然说这一步不是必需的，但是这样的话理解起来会更方便。

Line 26:这里设置了学习速率。

Line 27:我们要把两个数加起来，所以我们一次要输入两位字符。如此以来，我们的网络就需要两个输入。

Line 28:这是隐含层的大小，回来存储“携带位”。需要注意的是，它的大小比原理上所需的要大。自己尝试着调整一下这个值，然后看看它如何影响收敛速率。更高的隐含层维度会使训练变慢还是变快？更多或是更少的迭代次数？

Line 29:我们只是预测和的值，也就是一个数。如此，我们只需一个输出。

Line 33:这个权值矩阵连接了输入层与隐含层，如此它就有“imput_dim”行以及“hidden_dim”列（假如你不改参数的话就是2×16）。

Line 34:这个权值矩阵连接了隐含层与输出层，如此它就有“hidden_dim”行以及“output_dim”列（假如你不改参数的话就是16×1）。

Line 35:这个权值矩阵连接了前一时刻的隐含层与现在时刻的隐含层。它同样连接了当前时刻的隐含层与下一时刻的隐含层。如此以来，它就有隐含层维度大小（hidden_dim）的行与隐含层维度大小（hidden_dim）的列（假如你没有修改参数就是16×16）。

Line 37-39:这里存储权值更新。在我们积累了一些权值更新以后，我们再去更新权值。这里先放一放，稍后我们再详细讨论。

Line 42:我们迭代训练样例10000次。

Line 45:这里我们要随机生成一个在范围内的加法问题。所以我们生成一个在0到最大值一半之间的整数。如果我们允许网络的表示超过这个范围，那么把两个数加起来就有可能溢出（比如一个很大的数导致我们的位数不能表示）。所以说，我们只把加法要加的两个数字设定在小于最大值的一半。

Line 46:我们查找a_int对应的二进制表示，然后把它存进a里面。

Line 48:原理同45行。

Line 49:原理同46行。

Line 52:我们计算加法的正确结果。

Line 53:把正确结果转化为二进制表示。

Line 56:初始化一个空的二进制数组，用来存储神经网络的预测值（便于我们最后输出）。你也可以不这样做，但是我觉得这样使事情变得更符合直觉。

Line 58:重置误差值（这是我们使用的一种记录收敛的方式……可以参考之前关于反向传播与梯度下降的文章）

Line 60-61:这两个list会每个时刻不断的记录layer 2的导数值与layer 1的值。

Line 62:在0时刻是没有之前的隐含层的，所以我们初始化一个全为0的。

Line 65:这个循环是遍历二进制数字。

Line 68:X跟图片中的“layer_0”是一样的，X数组中的每个元素包含两个二进制数，其中一个来自a，一个来自b。它通过position变量从a，b中检索，从最右边往左检索。所以说，当position等于0时，就检索a最右边的一位和b最右边的一位。当position等于1时，就向左移一位。

Line 69:跟68行检索的方式一样，但是把值替代成了正确的结果（0或者1）。

Line 72:这里就是奥妙所在！一定一定一定要保证你理解这一行！！！为了建立隐含层，我们首先做了两件事。第一，我们从输入层传播到隐含层（np.dot(X,synapse_0)）。然后，我们从之前的隐含层传播到现在的隐含层（np.dot(prev_layer_1.synapse_h)）。在这里，layer_1_values[-1]就是取了最后一个存进去的隐含层，也就是之前的那个隐含层！然后我们把两个向量加起来！！！！然后再通过sigmoid函数。

那么，我们怎么结合之前的隐含层信息与现在的输入呢？当每个都被变量矩阵传播过以后，我们把信息加起来。

Line 75:这行看起来很眼熟吧？这跟之前的文章类似，它从隐含层传播到输出层，即输出一个预测值。

Line 78:计算一下预测误差（预测值与真实值的差）。

Line 79:这里我们把导数值存起来（上图中的芥末黄），即把每个时刻的导数值都保留着。

Line 80:计算误差的绝对值，并把它们加起来，这样我们就得到一个误差的标量（用来衡量传播）。我们最后会得到所有二进制位的误差的总和。

Line 86:将layer_1的值拷贝到另外一个数组里，这样我们就可以下一个时间使用这个值。

Line 90:我们已经完成了所有的正向传播，并且已经计算了输出层的导数，并将其存入在一个列表里了。现在我们需要做的就是反向传播，从最后一个时间点开始，反向一直到第一个。

Line 92:像之前那样，检索输入数据。

Line 93:从列表中取出当前的隐含层。

Line 94:从列表中取出前一个隐含层。

Line 97:从列表中取出当前输出层的误差。 

Line 99:这一行计算了当前隐含层的误差。通过当前之后一个时间点的误差和当前输出层的误差计算。

Line 102-104:我们已经有了反向传播中当前时刻的导数值，那么就可以生成权值更新的量了（但是还没真正的更新权值）。我们会在完成所有的反向传播以后再去真正的更新我们的权值矩阵，这是为什么呢？因为我们要用权值矩阵去做反向传播。如此以来，在完成所有反向传播以前，我们不能改变权值矩阵中的值。

Line 109-115:现在我们就已经完成了反向传播，得到了权值要更新的量，所以就赶快更新权值吧（别忘了重置update变量）！

Line 118-end:这里仅仅是一些输出日志，便于我们观察中间的计算过程与效果。
'''