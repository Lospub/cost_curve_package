# Cost Curve Project
Consider the task of identifying which of a set of (binary) classifiers is best for a task. Determining this most cost-effective classifier depends on both properties of the classifier (in particular, its True Positive Rate and True Negative Rate), and the properties of its environment (i.e., the population’s proportion of positive instances, as well as the cost for false positives and cost for false negatives).  We show that the Cost Curves approach provides an effective way to identify the best classifier for different environments, using both classifier properties (learned from a single training dataset) and user-input characteristics of the target environment.  This paper then describes a web app that implements this Cost Curve approach, allowing the user to enter properties of a set of classifiers, and be able to determine (and easily visualize) which classifier is best for each environment. We also compare this approach to the commonly used AUROC (Area Under the Receiver-Operator Curve) approach. </br >

For more details about cost curve: </br >

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;["Cost-Sensitive Classifier Evaluation Using Cost Curves"](https://webdocs.cs.ualberta.ca/~holte/Publications/flairs2011.pdf)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;["Cost curves: An improved method for visualizing classifier performance"](https://webdocs.cs.ualberta.ca/~holte/Publications/mlj2006.pdf)
