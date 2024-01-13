'''
Normalized Cost Curve 
'''
import matplotlib.pyplot as plt
import numpy as np
from sympy import symbols, solve
from sklearn import metrics
from scipy.spatial import ConvexHull

models_color = {}

def build_plot(x_lim, y_lim, x_label, y_label, title):
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
        if ppv is None:
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
    p=0
    for _, value in enumerate(y):
        if value == 1:
            p += 1

    ppv = p / len(y)
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
        if select_n is not None and select_n <= len(value):
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
    fpr = round(pair[0], 2)
    tpr = round(pair[1], 2)
    thr = round(pair[2], 2)
    return (fpr, tpr, thr)

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
               pos_misclass = 1, neg_misclass = 1, low_envelope=True, trivial=True, operating=False, classifiers=True, dominated=False,
               roc_line=False,  convex=True, roc_trivial=True, model_costs = None):
    '''
    String data
    
    Args:
        pair (tuple): the data pair
    
    Return:
        pair (tuple): the stringfied data pair
    '''

    if model_costs is None:
        model_costs = {}

    # identify number of plots based on input type
    fig1 = plt.figure(figsize=(12,5))
    ax = plt.subplot(1, 1, 1)

    cost = CostCurve(inputs, true_label, select_n, model_costs)

    # build cost curve
    if misclass:
        value, y_, sols = cost.cost_curve_with_misclass(ax, pos_prob, misclass_ratio, pos_misclass, neg_misclass, low_envelope, trivial, operating, classifiers, dominated)
    else:
        value, y_, sols = cost.cost_curve_without_misclass(ax, pos_prob, low_envelope, trivial, operating, classifiers, dominated)

    # build roc curve
    fig2 = plt.figure(figsize=(12,5))
    ax_ = plt.subplot(1, 1, 1)
    models = cost.models.copy()
    pairs = cost.pairs.copy()
    init_pair = cost.init_pair.copy()
    roc = roc_curve()
    roc.draw_roc_curve(ax_, models, pairs, init_pair, roc_line, convex, roc_trivial)

    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    return fig1, fig2, value, y_, sols

class CostCurve:
    def __init__(self, inputs, true_label = False, select_n = 10, model_costs = None):
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

        self.sols = {}
        self.optimal_pair = [None]*len(inputs.keys())
        self.model_costs = model_costs
        self.y_value = None
        self.x_value = None
        self.ppv = None
        self.misclass = None
        if self.model_costs is None:
            self.model_costs = {}
        if true_label:
            self.ppv, self.models, self.pairs, self.init_pair = read_label(inputs, select_n, self.ppv)
        else:
            self.models, self.pairs, self.init_pair = read_data(inputs, select_n)

    def cost_curve(self, ax, color, model, pair, index, low_envelope, operating, classifiers, dominated):
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
        # calculate_pair.append((0,0,0))
        # calculate_pair.append((1,1,1))
        # print(pair)
        fprs, tprs, thresholds = map(list, zip(*pairs))
        fprs_, tprs_, _ = map(list, zip(*calculate_pair))
        pc, nec, lines, slpoes, intercepts, records, lower_envelope_list, lower_envelope_pair_list, lower_round = ([] for i in range(9))

        if self.misclass:
            self.y_value = min((1-tpr)*self.x_value + fpr*(1-self.x_value) for tpr, fpr in zip(tprs_, fprs_))

        # compute a line in the cost space for each point in the roc space
        for fpr, tpr, threshold in zip(fprs, tprs, thresholds):
            slope = 1-tpr-fpr
            intercept = fpr
            lines.append((slope, intercept))
            slpoes.append(slope)
            intercepts.append(intercept)
            records.append(((slope, intercept), fpr, tpr, threshold))

        # inital the x value
        for i in np.arange(0, 1.01, 0.01):
            pc.append(i)

        # compute the lower envelope
        for x_value in pc:
            y_value = min([(slope * x_value + intercept) for slope, intercept in zip(slpoes, intercepts)])
            lower_envelope_pair_list.append((round(x_value, 2), round(y_value, 12)))
            lower_round.append((round(x_value, 2), round(y_value, 3)))
            lower_envelope_list.append(round(y_value, 12))

        for pair, fpr, tpr, slope, intercept in zip(pairs, fprs, tprs, slpoes, intercepts):

            # compute the cost curves
            for i in pc:
                cost = (1-tpr)*i + fpr*(1-i)
                nec.append(round(cost, 12))

            # display cost curve either with or not with domainted lines
            self.display_classifiers(ax, color, model, pair, index, dominated, pc, nec, lower_envelope_list, lower_envelope_pair_list, slope, intercept, classifiers)
            # ax.plot(pc, nec, linestyle = '-.', label = inital_pair_label(model, decimal_pair(pair)))
            #clear lists for next iteration
            nec.clear()

        # compute and display the operating range
        self.operating_range(ax, color, model, operating, lines)

        # display the lower envelope as a thicker black line
        self.display_lower_envelope(ax, color, model, low_envelope, pc, lower_envelope_list)

        # find intersections to build misclassified cost ratio range
        intersections = self.intersection_finder(slpoes, intercepts, lower_envelope_pair_list)
        sols = [(0,0,0)]
        thr = []
        for i in intersections:
            # [1-p(+)] ✖ FPR(α)  ✖ ratio( α ) ✖ C(–|+)+P(+)  ✖ C(–|+)✖  FNR(α) = y_value * total
            matched = [record for record in records if record[0] == i[1]]
            _fpr = matched[0][1]
            _tpr = matched[0][2]
            _threshold = matched[0][3]
            _index = fprs.index(_fpr)
            if _threshold not in thr:
                thr.append(_threshold)

            _matched = [record for record in records if record[0] == i[2]]
            _fpr_ = _matched[0][1]
            _threshold_ = _matched[0][3]
            _index_ = fprs.index(_fpr_)
            if _threshold_ not in thr:
                thr.append(_threshold_)

            x = symbols('x')
            # curve = (0.8*0.3*x+0.2*0.3)/(0.8+0.2*x) - i[0][1]
            curve = ((1-self.ppv)*_fpr*x+self.ppv*(1-_tpr))/(self.ppv+(1-self.ppv)*x) - i[0][1]
            sol = solve(curve)
            sols.append((round(sol[0], 2), pairs[_index][2], pairs[_index_][2]))
        sols.sort()
        self.sols[model] = []
        marked_thr = []
        for i in range(0,len(sols)-1):
            marked_thr.append(sols[i+1][1])
            item = {"THR":str(sols[i+1][1]), "ratio_from": format(sols[i][0], '.2f'), "ratio_to": format(sols[i+1][0], '.2f')}
            self.sols[model].append(item)
        left = set(thr) - set(marked_thr)
        last_item = {"THR":str(list(left)[0]), "ratio_from": format(sols[len(sols)-1][0], '.2f'), "ratio_to": 'inf'}
        self.sols[model].append(last_item)

    def display_classifiers(self, ax, color, model, pair, index, dominated, pc, nec, lower_envelope_list, lower_envelope_pair, slope, intercept, classifiers):
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
            if classifiers:
                # ideantify the optimal classifier
                if round(self.y_value, 12) == round(y, 12):
                    # if slope != 1 and slope != -1:
                    self.init_pair[index] = pair
                    self.optimal_pair[index] = [model, decimal_pair(pair), y]
                    label_pair = inital_pair_label(model, plot_pair)

                    # plot the optimal classifier
                    ax.plot(pc, nec, color=color, label = label_pair)

                    # else:
                    #     self.init_pair[index] = None
                    #     self.optimal_pair[index] = None
                else:
                    # display the curve other than the optimal classifier
                    if dominated:
                    #     if slope != 1 and slope != -1:
                        ax.plot(pc, nec, linestyle = '-.', label = inital_pair_label(model, plot_pair))
            else:
                if dominated:
                    # if slope != 1 and slope != -1:
                    ax.plot(pc, nec, linestyle = '-.', label = inital_pair_label(model, plot_pair))
        # plot without misclassification cost
        else:
            intersect = [value for value in nec if value in lower_envelope_list]
            intersects = list(dict.fromkeys(intersect))
            # if slope != 1 and slope != -1:

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
                curve = slope * x_value + intercept
                left_intersection_x = solve(curve - x_value, x_value)
                if len(left_intersection_x) != 0:
                    intersections_left.append(left_intersection_x)

                right_intersection_x = solve(curve - (- x_value + 1), x_value)
                if right_intersection_x != 0:
                    intersections_right.append(right_intersection_x)

            # draw left bouundary
            x_inter_left = min(x for x in intersections_left)
            name = "Operating Range " + model
            ax.plot([x_inter_left[0], x_inter_left[0]], [0, x_inter_left[0]], color=color, label = name, alpha=0.7, linewidth=3, linestyle="--")

            # draw right boundary
            x_inter_right = max(x for x in intersections_right)
            y_inter_right = -x_inter_right[0]+1
            count = 0
            for line in ax.get_lines():
                if line.get_label() == name:
                    count -= 1
                count += 1
            ax.plot([x_inter_right[0], x_inter_right[0]], [0, y_inter_right], ax.get_lines()[count].get_c(), alpha=0.7, linewidth=3, linestyle="--")
        else:
            return None

    def intersection_finder(self, slpoes, intercepts, lower_envelope_pair_list):
        x_value = symbols('x')
        length = len(slpoes)
        intersections = []
        intersect_r = []
        for i in range(length):
            curve_1 = slpoes[i] * x_value + intercepts[i]
            # _intersection_1 = solve(curve_1 - x_value, x_value)
            # _intersection_2 = solve(curve_1 - (-x_value+1), x_value)
            # _intersection_1_x = round(_intersection_1[0], 2)
            # _intersection_2_x = round(_intersection_2[0], 2)
            # intersections.append((_intersection_1_x, round( slpoes[i] *_intersection_1_x + intercepts[i], 12)))
            # intersections.append((_intersection_2_x, round( slpoes[i] *_intersection_2_x + intercepts[i], 12)))
            if i != length - 1:
                for j in range(i+1, length):
                    curve_2 = slpoes[j] * x_value + intercepts[j]
                    intersection = solve(curve_1 - curve_2, x_value)
                    intersection_x = round(float(intersection[0]), 2)
                    inter_y = min(slpoes[i] *intersection_x + intercepts[i], slpoes[j] *intersection_x + intercepts[j])
                    intersections.append([(intersection_x, round(inter_y, 12)), (slpoes[i], intercepts[i]), (slpoes[j], intercepts[j])])
        intersections.sort()
        for intersect in intersections:
            if intersect[0] in lower_envelope_pair_list:
                intersect_r.append(intersect)

        intersect_r.sort()
        return intersect_r

    def trivial_classification(self, ax, trivial):
        '''
        Draw trivial classifiers
        
        Args:
            ax (subplot)
            trivial (boolean): true - draw trivial classifier, false - not draw trivial classifier
        '''
        if trivial:
            ax.plot([0, 0.5, 1], [0, 0.5, 0], color="yellow", label="Trivial Classifier", linewidth=4)
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
            ax.plot(pc, lower_envelope_list, color=color, label="Lower Envelope "+model, linewidth=4, alpha = .6)

    def cost_curve_without_misclass(self, ax, pos_prob = 0.5, low_envelope = True, trivial = True, operating = True, classifiers = True, dominated = False):
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
        x_value = pos_prob
        self.x_value = x_value
        self.misclass = True

        self.trivial_classification(ax, trivial)
        for i in range(len(self.models)):
            model = self.models[i][0]
            pair = self.pairs[i].copy()
            color = next(ax._get_lines.prop_cycler)['color']
            models_color[model] = color
            self.cost_curve(ax, color, model, pair, i, low_envelope, operating, classifiers, dominated)
            y_value = self.y_value
            ax.plot(x_value, y_value, 'o', color=color, markersize = 10)

        value, y_ = self.optimal_classifier()
        build_plot([0.0, 1.0], [0.0, 0.5], "Probability Cost Function", "Normalized Expected Cost", "Normalized Cost Curve")

        return value, y_, self.sols

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
                if optimal is None and self.optimal_pair[i] is not None:
                    optimal = self.optimal_pair[i]
                elif optimal is not None and self.optimal_pair[i] is not None:
                    if optimal[2] > self.optimal_pair[i][2]:
                        optimal = self.optimal_pair[i]
        if optimal is not None:
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
        self.ppv = pos_prob
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
            color = next(ax._get_lines.prop_cycler)['color']
            models_color[model] = color
            self.cost_curve(ax, color, model, pair, i, low_envelope, operating, classifier, dominated)
            y_value = self.y_value
            ax.plot(x_value, y_value, 'o', color=color, markersize = 10)

        value, y_ = self.optimal_classifier()
        # build_plot([0.0, 1.0], [0.0, 0.5], "Probability Cost Function", "Normalized Expected Cost", "Normalized Cost Curve")

        build_plot([0.0, 1.0], [0.0, 0.5], "PC(+)", "Normalized Expected Cost", "Normalized Cost Curve")
        return value, y_, self.sols

class roc_curve:
    def draw_roc_curve(self, ax, models, pairs, init_pair, roc_line=True,  convex=False, roc_trivial=True):
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

            fprs, tprs, thre = map(list, zip(*pair))
            if convex:
                new_pairs = [tuple(x) for x in zip(fprs, tprs)]
                self.convex_hill(ax, model=model, pairs=new_pairs)

            # display roc line
            fprs, tprs, thre = map(list, zip(*pair))
            fprs = np.array(fprs)
            tprs = np.array(tprs)
            aucroc = metrics.auc(fprs, tprs)
            str_auroc = str(round(aucroc, 2))
            # for fpr, tpr in zip(fprs, tprs):
            #     print(fpr, tpr)
            #     print(metrics.auc([0.0,fpr,1.0], [0.0,tpr,1.0]))
            if roc_line:
                ax.plot(fprs, tprs, label = model, color=models_color[model], linestyle='--')

            # notify the ROC point which is currently displayed in cost curve
            if init_pair[i] != None:
                # display optimal roc points
                init_fpr = [init_pair[i][0]]
                init_tpr = [init_pair[i][1]]
                aucroc = metrics.auc([0.0,init_pair[i][0],1.0], [0.0,init_pair[i][1],1.0])
                str_auroc = str(round(aucroc, 2))
                label_pair = inital_pair_label(model, decimal_pair(init_pair[i]))
                # ax.plot(init_fpr, init_tpr, 'o', color=models_color[model], markersize = 10, label=label_pair+'(AUROC:'+str_auroc+")")

        # display trivial line
        self.trivial(ax, roc_trivial)

        # plot parameters
        build_plot(x_lim=[0.0, 1.0], y_lim=[0.0, 1.0], x_label="False positive Rate", y_label="True Positive Rate", title="ROC Curve")

    def trivial(self, ax, roc_trivial):
        '''
        Draw trivial classifiers
        
        Args:
            ax (subplot)
            trivial (boolean): true - draw trivial classifier, false - not draw trivial classifier
        '''
        if roc_trivial:
            ax.plot([0, 1], [0, 1], color='yellow', label = 'Trivial Classifier', linewidth=3)

    def convex_hill(self, ax, model, pairs):
        """
        Draw convex hill

        Args:
            ax (subplot)
            convex (bool, optional): true - draw convex hill, false - not draw convex hill.
            pairs (list): list of the pair data
        """
        points = np.array(pairs)
        fprs, tprs = map(list, zip(*pairs))
        # aucroc = metrics.auc(fprs, tprs)
        # str_auroc = str(round(aucroc, 2))
        hull = ConvexHull(points)
        count = 0
        for simplex in hull.simplices:
            if count == 0:
                ax.plot(points[simplex,0], points[simplex,1], color=models_color[model], linewidth=2.5,
                        label=model+" -- convex hull")
            ax.plot(points[simplex,0], points[simplex,1], color=models_color[model], linewidth=2.5)
            count += 1
