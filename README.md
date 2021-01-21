# multi-head-selft-attention-lstm
在sts数据集上用多头注意力机制上进行测试。 pytorch torchtext 代码简练，非常适合新手了解多头注意力机制的运作。不像transformer牵扯很多层这里仅仅用了 multi-head attention + one layer lstm
#参考了各位大神的demo
苏大神keras版本：

https://kexue.fm/archives/4765


github多头注意力模板，但是没有代码彻底实现，我这里增加了测试的例子
https://github.com/sakuranew/attention-pytorch

代码里面没有给出评估函数，但是loss下降都正常。用debug，以了解分析多头注意力的小demo，多头注意力的scale没有完全按照维度根号整除，简化了

#《Attention is All You Need》
在跟腾讯衍天实验室的一次合作中，写注意力模型，发现公式看上去很简单的东西甚至之前都完全看懂过代码，但换个场景可能就会比较懵逼。必须要代码去实现，并且不仅仅是一模一样的实现，一样的场景实现，而是把问题拆分，灵活分解后去实现才可能深入的理解问题，
在自己实现代码的过程中就理解了多头的概念，而曾经仅仅实现transformer的时候，对多头印象并不深刻，多头其实是一种优化概念，并没有严格的数学证明。而且，效果一般般，是基于自注意力机制提出来的提升。
