import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, solve
from sklearn import metrics

color_model = {"NB[0.3]":"#1f77b4", "NB[0.8]":"darkorange", "NB[0.5]":"green"}

def build_plot(x_lim, y_lim, x_label, y_label, title):
    """
    format the plot
    
    Args:
        x_lim (list): domain for x-axis
        y_lim (list): domain for y-axis
        x_label (string): the label name for x-axis
        y_label (string): the label name for y-axis
        title (string): the title for subplot
    """ 
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.legend()
    plt.title(title, fontsize=20)
    plt.show()


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
            models.append((key, select_n))
        else:
            num = len(value)
            models.append((key, num))
        pairs.append(value)
        
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

def inital_pair_label(model, plot_pair):
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
    
    return model+"["+str(plot_pair[2])+"]"

def curves(inputs, misclass=True, true_label = False, select_n = 10, pos_prob = 0.5, misclass_ratio=1, 
               pos_misclass = 1, neg_misclass = 1, low_envelope=True, trivial=True, operating=True, classifiers=True, dominated=False,
               roc_line=True,  convex=False, model_costs = {}):
    
    # identify number of plots based on input type
    fig1 = plt.figure(figsize=(12,5))
    ax = plt.subplot(1, 1, 1)
    
    
    cost = cost_curve(inputs, true_label, select_n, model_costs)
    
    # build cost curve
    if misclass:
        value, y_ = cost.cost_curve_with_misclass(ax, pos_prob, misclass_ratio, pos_misclass, neg_misclass, low_envelope, trivial, operating, classifiers, dominated)
    else:
        cost.cost_curve_without_misclass(ax, low_envelope, trivial, operating, classifiers, dominated)

    
    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)
    return fig1, value, y_
    
class cost_curve:
    def __init__(self, inputs, true_label = False, select_n = 10, model_costs = {}):
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
        self.model_costs = model_costs
        if true_label:
            self.ppv, self.models, self.pairs, self.init_pair = read_label(inputs, select_n, self.ppv)
        else:
            self.models, self.pairs, self.init_pair = read_data(inputs, select_n)
    
    def cost_curve(self, ax, color, model, pair, index, low_envelope, operating, classifiers, dominated, neg_misclass, pos_misclass):
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
        calculate_pair = pair.copy()
        
        # print(pairs)
        fprs, tprs, thresholds = map(list, zip(*pairs))
        fprs_, tprs_, thresholds = map(list, zip(*calculate_pair))
        pc, nec, lines, slpoes, intercepts, lower_envelope_list, lower_envelope_pair_list = ([] for i in range(7))
        
        if self.misclass:
            y_value = min(((1-tpr)*self.x_value*neg_misclass + fpr*(1-self.x_value)*pos_misclass) for tpr, fpr in zip(tprs_, fprs_)) + self.model_costs[model]
            trivial_ = [(0,0), (1,1)]
            trivial_value = min(((1-tpr)*self.x_value*neg_misclass + fpr*(1-self.x_value)*pos_misclass) for tpr, fpr in trivial_)
            # if y_value > trivial_value:
            #     self.y_value = trivial_value
            # else:
            self.y_value = y_value

        # compute a line in the cost space for each point in the roc space
        # for fpr, tpr in zip(fprs, tprs):
        #     slope = 1-tpr-fpr
        #     intercept = fpr
        #     lines.append((slope, intercept))
        #     slpoes.append(slope)
        #     intercepts.append(intercept)
            
        # inital the x value
        for i in np.arange(0, 1.01, 0.01):
            pc.append(i)

        # compute the lower envelope    
        for x_value in pc:
            y_value = min([((1-tpr)*(x_value*neg_misclass) + fpr*(1-x_value)*pos_misclass)  for fpr, tpr in zip(fprs, tprs)]) + self.model_costs[model]
            lower_envelope_pair_list.append((x_value, round(y_value, 12)))
            lower_envelope_list.append(round(y_value, 12))
        
        
        for pair, fpr, tpr in zip(pairs, fprs, tprs):
            
            # compute the cost curves
            for i in pc:
                cost = ((1-tpr)*(i*neg_misclass) + fpr*(1-i)*pos_misclass) + self.model_costs[model]
                nec.append(round(cost, 12))
            # print(nec)

            # display cost curve either with or not with domainted lines
            self.display_classifiers(ax, color, model, pair, index, dominated, pc, nec, lower_envelope_list, lower_envelope_pair_list, fpr, tpr, classifiers, neg_misclass, pos_misclass)

            # ax.plot(pc, nec, linestyle = '-', label = inital_pair_label(model, decimal_pair(pair)))
            #clear lists for next iteration
            nec.clear()
        
        # compute and display the operating range
        self.operating_range(ax, color, model, operating, lines)

        # display the lower envelope as a thicker black line
        self.display_lower_envelope(ax, color, model, low_envelope, pc, lower_envelope_list)
    
    def display_classifiers(self, ax, color, model, pair, index, dominated, pc, nec, lower_envelope_list, lower_envelope_pair, fpr, tpr, classifiers, neg_misclass, pos_misclass):
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
            y = ((1-tpr)*(self.x_value*neg_misclass) + fpr*(1-self.x_value)*pos_misclass) + self.model_costs[model]
            if classifiers:
                # ideantify the optimal classifier
                if round(self.y_value, 12) == round(y, 12):
                    
                    self.init_pair[index] = pair
                    self.optimal_pair[index] = [model, decimal_pair(pair), y]
                    print(plot_pair)
                    label_pair = inital_pair_label(model, plot_pair)
                    
                    # plot the optimal classifier 
                    ax.plot(pc, nec, color = color_model[label_pair], label = label_pair)
                    

                else:
                    # display the curve other than the optimal classifier
                    if dominated:
                        ax.plot(pc, nec, color = color_model[inital_pair_label(model, plot_pair)], linestyle = '-.',  label = inital_pair_label(model, plot_pair))
            else:
                if dominated:
                    ax.plot(pc, nec, color = color_model[inital_pair_label(model, plot_pair)], linestyle = '-.', label = inital_pair_label(model, plot_pair))         
        # plot without misclassification cost
        else:
            intersect = [value for value in nec if value in lower_envelope_list]
            intersects = list(dict.fromkeys(intersect))
                
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
                                ax.plot(pc, nec, color=color,label = inital_pair_label(model, plot_pair))
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
                                ax.plot(pc, nec, color=color, label = inital_pair_label(model, plot_pair))
                                is_draw = True
                                break
                    if not is_draw:
                        ax.plot(pc, nec, linestyle = '--', color=color, label = inital_pair_label(model, plot_pair))
                else:
                    ax.plot(pc, nec, linestyle = '--', color=color, label = inital_pair_label(model, plot_pair))
    
    def operating_range(self, ax, color, model, operating, lines):
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
                curve = (slope * x_value + intercept) * self.total_cost + self.model_costs[model]
                left_intersection_x = solve(curve - (x_value * self.total_cost), x_value)
                if len(left_intersection_x) != 0:
                    intersections_left.append(left_intersection_x)
                
                right_intersection_x = solve(curve - ((-x_value+1) * self.total_cost), x_value)
                if right_intersection_x != 0:
                    intersections_right.append(right_intersection_x)
            
            # draw left bouundary 
            # print(intersections_left)
            x_inter_left = min(x for x in intersections_left)
            if x_inter_left[0] > 0.5 or x_inter_left[0] < 0:
                x_inter_left = [0]
            # print(x_inter_left)
            name = "Operating Range " + model
            ax.plot([x_inter_left[0], x_inter_left[0]], [0, x_inter_left[0]*self.total_cost], color=color, label = name, alpha=0.7, linewidth=3, linestyle="--")
            
            # draw right boundary
            x_inter_right = max(x for x in intersections_right)
            if 0.5 > x_inter_right[0] or 1 < x_inter_right[0]:
                x_inter_right = [1]
            y_inter_right = (-x_inter_right[0]+1)*self.total_cost
            ax.plot([x_inter_right[0], x_inter_right[0]], [0, y_inter_right], color=color, alpha=0.7, linewidth=3, linestyle="--")
        else:
            return None
        
    def trivial_classification(self, ax, trivial,pos_misclass, neg_misclass):
        '''
        Draw trivial classifiers
        
        Args:
            ax (subplot)
            trivial (boolean): true - draw trivial classifier, false - not draw trivial classifier
        '''
        if trivial:
            # inital the x value
            pc = []
            for i in np.arange(0, 1.01, 0.01):
                pc.append(i)

            fprs = [0, 1]
            tprs = [0, 1]
            lower_envelope_list = []
            # compute the lower envelope
            # for fpr, tpr in zip(fprs, tprs):
                # lower_envelope_list = []
            for x_value in pc:
                y_value = min([((1-tpr)*(x_value*neg_misclass) + fpr*(1-x_value)*pos_misclass)  for fpr, tpr in zip(fprs, tprs)]) 
                    # y_value = (1-tpr)*(x_value*neg_misclass) + fpr*(1-x_value)*pos_misclass
                lower_envelope_list.append(round(y_value, 12))
            self.y_max = max(lower_envelope_list)
            ax.plot(pc, lower_envelope_list, color="yellow", label="Trivial Classifier", alpha=0.5,linewidth=4)
            # ax.plot(pc, lower_envelope_list[101:202], color="yellow", alpha=0.5,linewidth=4)
        else:
            return None
        
    def display_lower_envelope(self, ax, color, model, low_envelope, pc, lower_envelope_list):
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
            ax.plot(pc, lower_envelope_list, color=color, label="Lower Envelope "+model, linewidth=4, alpha = .5)
            
    def cost_curve_without_misclass(self, ax, low_envelope = True, trivial = True, operating = True, classifiers = True, dominated = False):
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
        for i in range(len(self.models)):
            model = self.models[i][0]
            pair = self.pairs[i].copy()
            color = next(ax._get_lines.prop_cycler)['color']
            self.cost_curve(ax, color, model, pair, i, low_envelope, operating, classifiers, dominated)
        self.trivial_classification(ax, trivial)
        build_plot([0.0, 1.0], [0.0, 0.5], "Probability Cost Function", "Expected Cost", "Real Cost Curve")
        
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
        # if len(self.model_costs) > 0:
        #     return (positive_probability * neg_misclass + self.model_costs[model]) / (positive_probability * neg_misclass + (1 - positive_probability) * pos_misclass + self.model_costs[model])
        # else:
        return (positive_probability * neg_misclass) / (positive_probability * neg_misclass + (1 - positive_probability) * pos_misclass)
    
    def optimal_classifier(self):
        optimal = self.optimal_pair[0]
        if len(self.optimal_pair) > 1:
            for i in range(1, len(self.optimal_pair)):
                if optimal == None and self.optimal_pair[i] != None:
                    optimal = self.optimal_pair[i]
                elif optimal != None and self.optimal_pair[i] != None:
                    if optimal[2] > self.optimal_pair[i][2]:
                        optimal = self.optimal_pair[i]
        if optimal != None:
            value = inital_pair_label(optimal[0],optimal[1])
            y_ = round(optimal[2], 2)
                
        else:
            value = 'Trivial Classifier'
            y_ = None
        return value, y_
    
    def cost_curve_with_misclass(self, ax, pos_prob = 0.5, misclass_ratio=1, pos_misclass = 1, neg_misclass = 1, low_envelope=True, trivial=True, operating=True, classifier=True, dominated=False):
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
        # x_value = self.normalization(pos_misclass, neg_misclass, pos_prob)
        self.x_value = pos_prob
        self.misclass = True
        self.total_cost = pos_prob * neg_misclass + (1 - pos_prob) * pos_misclass
        

        # total_cost = 0
        self.trivial_classification(ax, trivial, pos_misclass, neg_misclass)

        # display the misclassification point
        for i in range(len(self.models)):
            model = self.models[i][0]
            pair = self.pairs[i].copy()
            
            # if len(self.model_costs) > 0:
            #     total_cost = pos_prob * neg_misclass + (1 - pos_prob) * pos_misclass + self.model_costs[model]
            # else:
            #     total_cost = pos_prob * neg_misclass + (1 - pos_prob) * pos_misclass
            # if self.total_cost > total_cost:
            #     total_cost = self.total_cost
            color = next(ax._get_lines.prop_cycler)['color']
            self.cost_curve(ax, color, model, pair, i, low_envelope, operating, classifier, dominated, neg_misclass, pos_misclass)
        
            y_value = self.y_value
            # # print(y_value)
            ax.plot(pos_prob, y_value, 'o', color = color, markersize = 10, alpha=.8)
            # ax.plot(0.75, 0.25, 'o', color='#1f77b4', markersize = 10, alpha=.8)
            
        
        pc = []
        for i in np.arange(0, 1.01, 0.01):
            pc.append(i)

        fprs = [0, 1]
        tprs = [0, 1]
        lower_envelope_list = []
        # compute the lower envelope
        # for fpr, tpr in zip(fprs, tprs):
            # lower_envelope_list = []
        for x_value in pc:
            y_value = min([((1-tpr)*(x_value*neg_misclass) + fpr*(1-x_value)*pos_misclass)  for fpr, tpr in zip(fprs, tprs)]) 
                # y_value = (1-tpr)*(x_value*neg_misclass) + fpr*(1-x_value)*pos_misclass
            lower_envelope_list.append(round(y_value, 12))
        self.y_max = max(lower_envelope_list)
        max_model_cost = max(self.model_costs[model[0]] for model in self.models)
        value, y_ = self.optimal_classifier()
        build_plot([0.0, 1.0], [0.0, self.y_max], "P(+)", "Expected Cost", "Cost Curve")
        return value, y_
