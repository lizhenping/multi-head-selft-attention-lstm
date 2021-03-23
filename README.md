# multi-head-selft-attention-lstm
在sts数据集上用多头注意力机制上进行测试。   pytorch torchtext 代码简练，非常适合新手了解多头注意力机制的运作。  不像transformer牵扯很多层这里仅仅用了 multi-head attention + one layer lstm
#参考了各位大神的demo
苏大神keras版本：

https://kexue.fm/archives/4765


github多头注意力模板，但是没有代码彻底实现，我这里增加了测试的例子
https://github.com/sakuranew/attention-pytorch

代码里面没有给出评估函数，但是loss下降都正常。用debug，以了解分析多头注意力的小demo，多头注意力的scale没有完全按照维度根号整除，简化了。  

# 《Attention is All You Need》
在跟腾讯衍天实验室的一次合作中，写注意力模型，发现公式看上去很简单的东西甚至之前都完全看懂过代码，但换个场景可能就会比较懵逼。  
必须要代码去实现，并且不仅仅是一模一样的实现，一样的场景实现，而是把问题拆分，灵活分解后去实现才可能深入的理解问题。
在自己实现代码的过程中就理解了多头的概念，而曾经仅仅实现transformer的时候，对多头印象并不深刻，多头其实是一种优化概念，并没有严格的数学证明。  
而且，效果一般般，是基于自注意力机制提出来的提升。  
# kaggle比赛页面，开课吧
https://www.kaggle.com/c/kkb-repl4nlp-assignment0/overview

比赛页面，清华大神发布的代码，写的蛮有趣的。是个学习transformers框架的的好例子
https://www.kaggle.com/barcarum/kkb-repl4nlp-a0-sbert-linear?scriptVersionId=52326354


同时里面提供了torchtext直接调用模型的例子
https://github.com/muralikrishnasn/semantic_similarity


这里安利下开课吧，国内培训的结合工程学术最好的，贪心有点偏向理论，或者单单偏向工程。  
当然都是很好的培训课程，开课吧的课程真的是把理论跟实践结合的非常好的，但是这也不是没有坏处，导致的结果就是学起来非常耗时。  
每个知识点，学习来都很慢。 
