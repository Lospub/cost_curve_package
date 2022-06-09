import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, solve
from sklearn import metrics
from scipy.spatial import ConvexHull
import seaborn as sn
import pandas as pd

# positive
mu, sigma = 10, 15 # mean and standard deviation
s_p = np.random.normal(mu, sigma, 500)

# negative
mu, sigma = -5, 10 # mean and standard deviation
s_n = np.random.normal(mu, sigma, 500)

def build_plot(ax, x_lim, y_lim, x_label, y_label, title):
    """
    format the subplot
    
    Args:
        ax (int): identify the subplot need to be formatted
        x_lim (list): domain for x-axis
        y_lim (list): domain for y-axis
        x_label (string): the label name for x-axis
        y_label (string): the label name for y-axis
        title (string): the title for subplot
    """
    if ax == 5:
        plt.subplot(1, 2, 1)
    elif ax == 6:
        plt.subplot(1, 2, 2)
        
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

def read_label(inputs, select_n, ppv):
    '''
    Read the type 2(True_pred) data and store it to models and pairs
    
    Args:
        inputs (dictionary): input data 
        select_n (int): the number of the selected thresholds.
        ppv (float): percentage of positive class
    
    Return:
        ppv (float): percentage of positive class, 
        models (list): list of model names, 
        pairs (list): list of tuple (false positive rate, true positive rate, threshold), 
        init_pair (list): list of none
    '''
    models, pairs = ([] for i in range(2))
    
    # Calculate fpr, tpr for each model
    for key in inputs.keys():
        value = inputs[key]
        y, scores = map(list, zip(*value))
        array_y = np.array(y)
        array_scores = np.array(scores)
        fpr, tpr, thresholds = metrics.roc_curve(array_y, array_scores)
        
        # calculate ppv if necessary
        if ppv == None:
            ppv = calculate_ppv(y)
        pair = [tuple(x) for x in zip(fpr, tpr, thresholds)]
        
        # select n thresholds 
        if select_n <= len(thresholds):
            index = np.linspace(0, len(thresholds)-1, select_n, dtype=int)
            pair = [pair[i] for i in index]
        else:
            select_n = len(thresholds)
        
        pairs.append(pair)
        models.append((key, select_n))
        init_pair = [None]*len(models)
        
    return ppv, models, pairs, init_pair

def calculate_ppv(y):
    '''
    Calculate positive probability rate
    
    Args:
        y (list): list of true value
        
    Return:
        ppv (float): percentage of positive class
    '''
    P=0
    for i in range(len(y)): 
        if y[i] == 1:
            P += 1
            
    ppv = P / len(y)
    ppv_rounded = round(ppv, 2)
    
    return ppv_rounded

def read_data(inputs, select_n):
    '''
    Read the type 1(FPR_TPR) data and store it to models and pairs
    
    Args:
        inputs (dictionary): input data 
        select_n (int): the number of the selected thresholds.
    
    Return:
        models (list): list of model names, 
        pairs (list): list of tuple (false positive rate, true positive rate, threshold), 
        init_pair (list): list of none
    '''
    models, pairs = ([] for i in range(2))
    for key in inputs.keys():
        value = inputs[key]
        
        # select n thresholds
        if select_n != None and select_n <= len(value):
            index = np.linspace(0, len(value)-1, select_n, dtype=int)
            value = [value[i] for i in index]
        else:
            select_n = len(value)
        
        pairs.append(value)
        models.append((key, select_n))
        init_pair = [None]*len(models)
    return models, pairs, init_pair

def decimal_pair(pair):
    '''
    Round data
    
    Args:
        pair (tuple): the data pair
    
    Return:
        (FPR, TPR, THR): the rounded data pair
    '''
    FPR = round(pair[0], 2)
    TPR = round(pair[1], 2)
    THR = round(pair[2], 2)
    
    return (FPR, TPR, THR)

def inital_pair_label(plot_pair):
    '''
    String data
    
    Args:
        pair (tuple): the data pair
    
    Return:
        pair (tuple): the stringfied data pair
    '''
    pair = list(plot_pair)
    pair[0] = "FPR: "+str(plot_pair[0])
    pair[1] = "TPR: "+str(plot_pair[1])
    pair[2] = "THR: "+str(plot_pair[2])
    
    return tuple(pair)

def curves(inputs, misclass=True, true_label = False, select_n = 10, pos_prob = 0.5, misclass_ratio=1, 
               pos_misclass = 1, neg_misclass = 1, low_envelope=True, trivial=True, operating=True, classifiers=True, dominated=False,
               roc_line=True,  convex=False):
    
    # identify number of plots based on input type
    plt.figure(figsize=(15,5))
    ax_ = plt.subplot(1, 2, 1)
    ax = plt.subplot(1, 2, 2)
    ax_num = 6
    ax_num_ = 5
    
    cost = cost_curve(inputs, true_label, select_n)
    
    # build cost curve
    if misclass:
        cost.cost_curve_with_misclass(ax, ax_num, pos_prob, misclass_ratio, pos_misclass, neg_misclass, low_envelope, trivial, operating, classifiers, dominated)
    else:
        cost.cost_curve_without_misclass(ax, ax_num, low_envelope, trivial, operating, classifiers, dominated)
    
    # build roc curve  
    models = cost.models.copy()
    pairs = cost.pairs.copy()
    init_pair = cost.init_pair.copy()
    roc = roc_curve()
    roc.draw_roc_curve(ax_, ax_num_, models, pairs, init_pair, trivial, roc_line, convex)
    
    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)
    plt.show()
    
class cost_curve:
    def __init__(self, inputs, true_label = False, select_n = 10):
        '''
        read models and pairs from input dictionary
        
        Args:
            inputs (dictionary): the input data. It can be two types: 
                type 1: 
                    The dictionary of FPR(false positive rate)-TPR(true positive rate)-THR(threshold). The key of dictionary is the name of prediction model and the value is pairs of fpr-tpr.
                    Example:
                        dict_1 = {'model_1':[(FPR1, TPR1, THR1), (FPR2, TPR2, THR2), (FPR3, TPR3, THR3), ...], 'model_2':[(FPR1, TPR1, THR1), (FPR2, TPR2, THR2), (FPR3, TPR3, THR3), ...], ...}
                type 2: 
                    The dictionary of True_predicted probability:
                    Example:
                        dict_1 = {'model_1':[(TRUE1, PRED1), (TRUE2, PRED2), (TRUE3, PRED2), ...], 'model_2':[(TRUE1, PRED1), (TRUE2, PRED2), (TRUE3, PRED2), ...], ...}
            true_label (boolean): a boolean value used to notified the input type.
                False: The input value is type 1
                True: The input value is type 2.
            select_n (int): the number of the selected thresholds. The defult value is 10. None for selected all thresholds
        '''
        # self.ppv: positive porobability(defult as first model in for multiple models)
        # self.optimal_pair: the list used to store the model name, its optimal classifier and its normalized expected cost
        # self.models: The list of all model names
        # sekf.pairs: The list of FPR-TPR-THR paris
        # self.init_pair: the list used to store the optimal classifier for each model
        
        self.ppv = None
        self.optimal_pair = [None]*len(inputs.keys())
        if true_label:
            self.ppv, self.models, self.pairs, self.init_pair = read_label(inputs, select_n, self.ppv)
        else:
            self.models, self.pairs, self.init_pair = read_data(inputs, select_n)
    
    def cost_curve(self, ax, model, pair, index, low_envelope, operating, classifiers, dominated):
        '''
        build the cost curve plot
        
        Args:
            ax (subplot)
            model (string): the current model name
            pair (list): the list of data pair used to build plot
            index (int): the index for the model in input data dictionary
            low_envelop (boolean): 
                        true - draw lower envelop, false - not draw lower envelop
            operating (boolean): 
                        true - display operating range, false - not display operating range
            classifiers (boolean): 
                        true - display the classifiers which used to build the lower envelope when displaying plot without misclassification cost, 
                                display the optimal classifier when displaying plot with misclassification cost 
                        false - display only the lower envelop
            domainted (boolean): 
                        true - display the classifier is outerperformed by another when displaying plot without misclassification cost, 
                            display all classifiers excapt optimal classifier when displaying plot with misclassification cost
                        false - not display the classifier is outerperformed by another when displaying plot without misclassification cost,
                            not display all classifiers excapt optimal classifier when displaying plot with misclassification cost
        '''
        pairs = pair.copy()
        pairs.append((0,0,0))
        pairs.append((1,1,1))
        fprs, tprs, thresholds = map(list, zip(*pairs))
        pc, nec, lines, slpoes, intercepts, lower_envelope_list, lower_envelope_pair_list = ([] for i in range(7))
        
        if self.misclass:
            self.y_value = min((1-tpr)*self.x_value + fpr*(1-self.x_value) for tpr, fpr in zip(tprs, fprs))

        # compute a line in the cost space for each point in the roc space
        for fpr, tpr in zip(fprs, tprs):
            slope = 1-tpr-fpr
            intercept = fpr
            lines.append((slope, intercept))
            slpoes.append(slope)
            intercepts.append(intercept)
            
        # inital the x value
        for i in np.arange(0, 1.01, 0.01):
            pc.append(i)

        # compute the lower envelope    
        for x_value in pc:
            y_value = min([(slope * x_value + intercept) for slope, intercept in zip(slpoes, intercepts)])
            lower_envelope_pair_list.append((x_value, round(y_value, 12)))
            lower_envelope_list.append(round(y_value, 12))
        
        if classifiers:
            for pair, fpr, tpr, slope, intercept in zip(pairs, fprs, tprs, slpoes, intercepts):
                
                # compute the cost curves
                for i in pc:
                    cost = (1-tpr)*i + fpr*(1-i)
                    nec.append(round(cost, 12))

                # display cost curve either with or not with domainted lines
                self.display_classifiers(ax, model, pair, index, dominated, pc, nec, lower_envelope_list, lower_envelope_pair_list, slope, intercept)

                #clear lists for next iteration
                nec.clear()
        
        # compute and display the operating range
        self.operating_range(ax, operating, lines)

        # display the lower envelope as a thicker black line
        self.display_lower_envelope(ax, model, low_envelope, pc, lower_envelope_list)
    
    def display_classifiers(self, ax, model, pair, index, dominated, pc, nec, lower_envelope_list, lower_envelope_pair, slope, intercept):
        """
        This function is used to build the specific cost curve line in the plot.

        Args:
            ax (subplot)
            model (string): the current model name
            pair (tuple): the data pair used to build cost curve line in the plot
            index (int): the index for the model in input data dictionary
            domainted (boolean): 
                        true - display the classifier is outerperformed by another when displaying plot without misclassification cost, 
                            display all classifiers excapt optimal classifier when displaying plot with misclassification cost
                        false - not display the classifier is outerperformed by another when displaying plot without misclassification cost,
                            not display all classifiers excapt optimal classifier when displaying plot with misclassification cost
            pc (list): the list of probability cost
            nec (list): the list of normalized expected cost
            lower_envelope_list (list): the list used to identified lower envelope
            slope (float): the slope of the cost curve
            intercept (float): the intercept of the cost curve
        """
        # round the data pair
        plot_pair = decimal_pair(pair)
        
        # plot with misclassification cots
        if self.misclass:
            y = slope * self.x_value + intercept
            
            # ideantify the optimal classifier
            if round(self.y_value, 12) == round(y, 12):
                if slope != 1 and slope != -1:
                    self.init_pair[index] = pair
                    self.optimal_pair[index] = [model, pair, y]
                    label_pair = inital_pair_label(plot_pair)
                    
                    # plot the optimal classifier 
                    ax.plot(pc, nec, label = label_pair)
                
                else:
                    self.init_pair[index] = None
                    self.optimal_pair[index] = None
            else:
                # display the curve other than the optimal classifier
                if dominated:
                    if slope != 1 and slope != -1:
                        ax.plot(pc, nec, linestyle = '-.', label = inital_pair_label(plot_pair))
                        
        # plot without misclassification cost
        else:
            intersect = [value for value in nec if value in lower_envelope_list]
            intersects = list(dict.fromkeys(intersect))
            if slope != 1 and slope != -1:
                
                # display the classifier is outerperformed by another
                if not dominated:
                    if len(intersect) != 0:
                        is_draw = False
                        pc_x = []
                        lower_x = []
                        for i in intersects:
                            if is_draw:
                                break
                            index_list_pc = [ item for item in range(len(nec)) if nec[item] == i ]
                            index_list_lower = [ item for item in range(len(lower_envelope_list)) if lower_envelope_list[item] == i ]
                            for index_ in index_list_pc:
                                pc_x.append(pc[index_])
                            for j in index_list_lower:
                                lower_x.append(lower_envelope_pair[j][0])
                            for x in lower_x:
                                if x in pc_x:
                                    ax.plot(pc, nec, label = inital_pair_label(plot_pair))
                                    is_draw = True
                                    break
                            
                
                # display all classifiers
                else:
                    if len(intersect) != 0:
                        is_draw = False
                        pc_x = []
                        lower_x = []
                        for i in intersects:
                            if is_draw:
                                break
                            index_list_pc = [ item for item in range(len(nec)) if nec[item] == i ]
                            index_list_lower = [ item for item in range(len(lower_envelope_list)) if lower_envelope_list[item] == i ]
                            for index_ in index_list_pc:
                                pc_x.append(pc[index_])
                            for j in index_list_lower:
                                lower_x.append(lower_envelope_pair[j][0])
                            for x in pc_x:
                                if x in lower_x:
                                    ax.plot(pc, nec, label = inital_pair_label(plot_pair))
                                    is_draw = True
                                    break
                        if not is_draw:
                            ax.plot(pc, nec, linestyle = '--', label = inital_pair_label(plot_pair))
                    else:
                        ax.plot(pc, nec, linestyle = '--', label = inital_pair_label(plot_pair))
    
    def operating_range(self, ax, operating, lines):
        '''
        Draw operating range
        
        Args:
            ax (subplot)
            operating (boolean): true - display operating range, false - not display operating range
            lines (list): the list with slopes and intercepts for all classifiers
        '''
        if operating:
            x_value = symbols('x')
            intersections_left = []
            intersections_right = []
            
            # find range boundary
            for slope, intercept in lines:
                curve = slope * x_value + intercept
                left_intersection_x = solve(curve - x_value, x_value)
                if len(left_intersection_x) != 0:
                    intersections_left.append(left_intersection_x)
                
                right_intersection_x = solve(curve - (-x_value+1), x_value)
                if right_intersection_x != 0:
                    intersections_right.append(right_intersection_x)
            
            # draw left bouundary 
            x_inter_left = min(x for x in intersections_left)
            ax.plot([x_inter_left[0], x_inter_left[0]], [0, x_inter_left[0]], color='red', label="Operating Range")
            
            # draw right boundary
            x_inter_right = max(x for x in intersections_right)
            y_inter_right = -x_inter_right[0]+1
            ax.plot([x_inter_right[0], x_inter_right[0]], [0, y_inter_right], color='red')
        else:
            return None
        
    def trivial_classification(self, ax, trivial):
        '''
        Draw trivial classifiers
        
        Args:
            ax (subplot)
            trivial (boolean): true - draw trivial classifier, false - not draw trivial classifier
        '''
        if trivial:
            ax.plot([0, 0.5, 1], [0, 0.5, 0], color="yellow", label="Trivial Classifier")
        else:
            return None
        
    def display_lower_envelope(self, ax, model, low_envelope, pc, lower_envelope_list):
        '''
        Draw lower envelope
        
        Args:
            ax (subplot)
            model (string): the current model name
            low_envelop (boolean): true - draw lower envelop, false - not draw lower envelop
            pc (list): the list of probability cost
            lower_envelope_list (list): the list used to identified lower envelope
        '''
        if low_envelope:
            ax.plot(pc, lower_envelope_list,  label="Lower Envelope "+model)
            
    def cost_curve_without_misclass(self, ax, ax_num, low_envelope = True, trivial = True, operating = True, classifiers = True, dominated = False):
        """
        generate cost curve plot without misclassification cost

        Args:
            ax (subplot)
            ax_num (int)
            low_envelop (boolean): 
                        true - draw lower envelop, false - not draw lower envelop
            trivial (boolean): 
                        true - draw trivial classifier, false - not draw trivial classifier
            operating (boolean): 
                        true - display operating range, false - not display operating range
            classifiers (boolean): 
                        true - display the classifiers which used to build the lower envelope 
                        false - display only the lower envelop
            domainted (boolean): 
                        true - display the classifier is outerperformed by another
                        false - not display the classifier is outerperformed by another
        """
        self.misclass = False
        self.trivial_classification(ax, trivial)
        for i in range(len(self.models)):
            model = self.models[i][0]
            pair = self.pairs[i].copy()
            self.cost_curve(ax, model, pair, i, low_envelope, operating, classifiers, dominated)
        build_plot(ax_num, [0.0, 1.0], [0.0, 0.5], "Probability Cost Function", "Normalized Expected Cost", "Cost Curve")
        
    def normalization(self, pos_misclass, neg_misclass, positive_probability):
        """
        calsulate the normalized expected cost with misclassification: p(+) * C(-|+) / (p(+) * C(-|+) + (1 - p(+)) * C(+|-))
        
        Args:
            pos_misclass (float): C(+|-) the cost of misclassifying a negative example 
            neg_misclass (float): C(-|+) the cost of misclassifying a positive example 
            positive_probability (float): p(+) positive porobability

        Returns:
            (float): the normalized expected cost
        """
        return (positive_probability * neg_misclass) / (positive_probability * neg_misclass + (1 - positive_probability) * pos_misclass)
    
    def cost_curve_with_misclass(self, ax, ax_num, pos_prob = 0.5, misclass_ratio=1, pos_misclass = 1, neg_misclass = 1, low_envelope=True, trivial=True, operating=True, classifier=True, dominated=False):
        """
        generate cost curve plot with misclassification cost

        Args:
            ax (subplot)
            ax_num (int)
            pos_prob (float, optional): p(+) positive porobability. Defaults to 0.5.
            misclass_ratio (int, optional): Positive to negative misclassification ratio. Defaults to 1.
            low_envelope (bool, optional): 
                    true - draw lower envelop, false - not draw lower envelop. 
                    Defaults to True.
            trivial (bool, optional): 
                    true - draw trivial classifier, false - not draw trivial classifier.
                    Defaults to True.
            operating (bool, optional): 
                    true - display operating range, false - not display operating range. 
                    Defaults to True.
            classifier (bool, optional): 
                    true - display the optimal classifier, false - display only the lower envelop. 
                    Defaults to True.
            dominated (bool, optional): 
                    true - display all classifiers excapt optimal classifier, false - not display all classifiers excapt optimal classifier. 
                    Defaults to False.
        """
        # identify the positive to negative misclassification and negative to positive misclassification 
        if misclass_ratio != 1:
            neg_misclass = misclass_ratio
            pos_misclass = 1
        
        # calculate optimal cost with misclassification    
        x_value = self.normalization(pos_misclass, neg_misclass, pos_prob)
        self.x_value = x_value
        self.misclass = True
        
        # display the misclassification point
        self.trivial_classification(ax, trivial)
        for i in range(len(self.models)):
            model = self.models[i][0]
            pair = self.pairs[i].copy()
            self.cost_curve(ax, model, pair, i, low_envelope, operating, classifier, dominated)
            y_value = self.y_value
            ax.plot(x_value, y_value, 'o')
            ax.plot([x_value, x_value], [0, y_value], '--')
            ax.plot([0, x_value], [y_value, y_value], '--')

        build_plot(ax_num, [0.0, 1.0], [0.0, 0.5], "Probability Cost Function", "Normalized Expected Cost", "Cost Curve")
        
class roc_curve:
    def draw_roc_curve(self, ax, ax_num, models, pairs, init_pair, trivial=True, roc_line=True,  convex=False):
        """
        generate the roc curve plot

        Args:
            ax (subplot)
            ax_num (int)
            models (list): list of model names, 
            pairs (list): list of tuple (false positive rate, true positive rate, threshold), 
            init_pair (list): list used to store the optimal classifier for each model
            trivial (bool, optional): true - draw trivial classifier, false - not draw trivial classifier. Defaults to True.
            roc_line (bool, optional): true - draw roc curve line, false not draw roc curve line. Defaults to True.
            convex (bool, optional): true - draw convex hill, false - not draw convex hill. Defaults to False.
        """
        
        for i in range(len(models)):
            model = models[i][0]
            pair = pairs[i].copy()
            pair.append((0,0,0))
            pair.append((1,1,1))
            pair.sort()
            pair_copy = pair.copy()
        
            # notify the ROC point which is currently displayed in cost curve
            if init_pair[i] != None:
                pair_copy.remove(init_pair[i])
                fpr, tpr, thresholds = map(list, zip(*pair_copy))
                ax.plot(fpr, tpr, 'o')

                # display optimal roc points
                init_fpr = [init_pair[i][0]]
                init_tpr = [init_pair[i][1]]
                label_pair = inital_pair_label(decimal_pair(init_pair[i]))
                ax.plot(init_fpr, init_tpr, 'or', markersize = 10, label=label_pair)

            else:
                fpr, tpr, thresholds = map(list, zip(*pair_copy))
                ax.plot(fpr, tpr, 'o')

            # display roc line
            fprs, tprs, thre = map(list, zip(*pair))
            if roc_line:
                ax.plot(fprs, tprs, label = model)
            
            # display convex hill
            new_pairs = [tuple(x) for x in zip(fprs, tprs)]
            self.convex_hill(ax, convex, new_pairs)
        
        # display trivial line
        self.trivial(ax, trivial)

        # plot parameters
        build_plot(ax_num, x_lim=[0.0, 1.0], y_lim=[0.0, 1.0], x_label="False positive Rate", y_label="True Positive Rate", title="ROC Curve")
    
    def trivial(selff, ax, trivial):
        '''
        Draw trivial classifiers
        
        Args:
            ax (subplot)
            trivial (boolean): true - draw trivial classifier, false - not draw trivial classifier
        '''
        if trivial:
            ax.plot([0, 1], [0, 1], color='yellow', label = 'Trivial Classifier')
    
    def convex_hill(self, ax, convex, pairs):
        """
        Draw convex hill

        Args:
            ax (subplot)
            convex (bool, optional): true - draw convex hill, false - not draw convex hill.
            pairs (list): list of the pair data
        """
        if convex:
            points = np.array(pairs)
            hull = ConvexHull(points)
            for simplex in hull.simplices:
                ax.plot(points[simplex,0], points[simplex,1], 'red')
                
