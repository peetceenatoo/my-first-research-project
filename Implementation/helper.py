# Import libraries
import numpy as np
from tqdm.notebook import tqdm
import numpy as np
import math
import pickle

# Function to pre-calculate the propensity scores
def calculate_propensities(n_users, n_items, trainfilename, gammas=[1.5, 2, 2.5, 3], normalize=True):

    # Init dictionaries
    propensities = dict()
    Ni = dict()

    # Compute frequencies of items in the training set
    trainset = np.load(trainfilename)
    for i in trainset['item_id']:
        if i in Ni:
            Ni[i] += 1
        else:
            Ni[i] = 1
    del trainset

    # Initialize propensities dictionary
    for gamma in gammas:
        propensities[gamma] = np.zeros((n_users,n_items))

    # Compute propensities according to the formula
    for theitem in range(n_items):
        if theitem not in Ni:
            continue
        for gamma in gammas:
            propensities[gamma][:,theitem] =  np.power(Ni[theitem], (gamma + 1) / 2.0)

    # Normalize propensities if required
    if normalize:
        for gamma in gammas:
            propensities[gamma] /= propensities[gamma].max()

    # Return
    return propensities

def eq(infilename, infilename_neg, trainfilename, propensities, K=1):

    # Read pickles
    infile = open(infilename, 'rb')
    infile_neg = open(infilename_neg, 'rb')
    P = pickle.load(infile)
    infile.close()
    P_neg = pickle.load(infile_neg)
    infile_neg.close()
    NUM_NEGATIVES = P["num_negatives"]
    
    # Merge P and P_neg
    for theuser in P["users"]:
        neg_items = list(P_neg["user_items"][theuser][NUM_NEGATIVES:])
        neg_scores = list(P_neg["results"][theuser][NUM_NEGATIVES:])
        P["user_items"][theuser] = list(neg_items) + list(P["user_items"][theuser][NUM_NEGATIVES:])
        P["results"][theuser] = list(neg_scores) + list(P["results"][theuser][NUM_NEGATIVES:])
    
    Zui = dict()
    Ni = dict()
    
    # Compute frequencies of items in the training set
    trainset = np.load(trainfilename)
    for i in trainset['item_id']:
        if i in Ni:
            Ni[i] += 1
        else:
            Ni[i] = 1
    del trainset

    # Count #users with non-zero item frequencies
    nonzero_user_count = 0
    for theuser in P["users"]:
        pos_items = P["user_items"][theuser][len(P_neg["user_items"][theuser][NUM_NEGATIVES:]):]
        for pos_item in pos_items:
            if pos_item in Ni:
                nonzero_user_count += 1
                break
    
    # Compute recommendations for each user
    for theuser in P["users"]:
        all_scores = np.array(P["results"][theuser])
        pos_items = P["user_items"][theuser][len(P_neg["user_items"][theuser][NUM_NEGATIVES:]):]
        pos_scores = P["results"][theuser][len(P_neg["results"][theuser][NUM_NEGATIVES:]):]
        for i, pos_item in enumerate(pos_items):
            pos_score = pos_scores[i]
            Zui[(theuser, pos_item)] = float(np.sum(all_scores > pos_score))

    
    # Calculate per-user scores
    sum_user_auc = 0.0
    sum_user_recall = 0.0
    for theuser in P["users"]:
        numerator_auc = 0.0
        numerator_recall = 0.0
        denominator = 0.0

        for theitem in P["user_items"][theuser][len(P_neg["user_items"][theuser][NUM_NEGATIVES:]):]:
            # Skip items with null frequency
            if theitem not in Ni:
                continue
            
            # Load propensity score
            pui = propensities[theuser][theitem]

            # Add things to be summed for each item
            numerator_auc += (1 - Zui[(theuser, theitem)] / len(P["user_items"][theuser])) / pui

            # Add things for recall
            if Zui[(theuser, theitem)] < K:
                numerator_recall += 1.0 / pui

            # Increment denominator that the sum must be divided by
            denominator += 1 / pui
                
        # If there was at least one item for the user, count the user and sum the results
        if denominator > 0:
            sum_user_auc += numerator_auc / denominator
            sum_user_recall += numerator_recall / denominator 

    # Return
    return {
        "auc"       : sum_user_auc / nonzero_user_count,
        "recall"    : sum_user_recall / nonzero_user_count,
        "bias"      : -1,
        "concentration" : -1
    }

def aoa(infilename, infilename_neg, trainfilename, K=1):

    # Read pickles
    infile = open(infilename, 'rb')
    infile_neg = open(infilename_neg, 'rb')
    P = pickle.load(infile)
    infile.close()
    P_neg = pickle.load(infile_neg)
    infile_neg.close()
    NUM_NEGATIVES = P["num_negatives"]
    
    # Merge P and P_neg
    for theuser in P["users"]:
        neg_items = list(P_neg["user_items"][theuser][NUM_NEGATIVES:])
        neg_scores = list(P_neg["results"][theuser][NUM_NEGATIVES:])
        P["user_items"][theuser] = list(neg_items) + list(P["user_items"][theuser][NUM_NEGATIVES:])
        P["results"][theuser] = list(neg_scores) + list(P["results"][theuser][NUM_NEGATIVES:])
    
    Zui = dict()
    Ni = dict()

    # Compute frequencies of items in the training set
    trainset = np.load(trainfilename)
    for i in trainset['item_id']:
        if i in Ni:
            Ni[i] += 1
        else:
            Ni[i] = 1
    del trainset
    
    # Count #users with non-zero item frequencies
    nonzero_user_count = 0
    for theuser in P["users"]:
        pos_items = P["user_items"][theuser][len(P_neg["user_items"][theuser][NUM_NEGATIVES:]):]
        for pos_item in pos_items:
            if pos_item in Ni:
                nonzero_user_count += 1
                break
    
    # Compute recommendations for each user
    for theuser in P["users"]:
        all_scores = np.array(P["results"][theuser])
        pos_items = P["user_items"][theuser][len(P_neg["user_items"][theuser][NUM_NEGATIVES:]):]
        pos_scores = P["results"][theuser][len(P_neg["results"][theuser][NUM_NEGATIVES:]):]
        for i, pos_item in enumerate(pos_items):
            pos_score = pos_scores[i]
            Zui[(theuser, pos_item)] = float(np.sum(all_scores > pos_score))

    # Calculate per-user scores
    sum_user_auc = 0.0
    sum_user_recall = 0.0
    for theuser in P["users"]:
        numerator_auc = 0.0
        numerator_recall = 0.0
        denominator = 0.0

        for theitem in P["user_items"][theuser][len(P_neg["user_items"][theuser][NUM_NEGATIVES:]):]:
            # Skip items with null frequency
            if theitem not in Ni:
                continue

            # Add things to be summed for each item
            numerator_auc += (1 - Zui[(theuser, theitem)] / len(P["user_items"][theuser]))

            # Add things for recall
            if Zui[(theuser, theitem)] < K:
                numerator_recall += 1.0
                
            # Increment denominator that the sum must be divided by
            denominator += 1 

        # If there was at least one item for the user, count the user and sum the results
        if denominator > 0:
            sum_user_auc += numerator_auc / denominator
            sum_user_recall += numerator_recall / denominator

    # Return
    return {
        "auc"       : sum_user_auc / nonzero_user_count,
        "recall"    : sum_user_recall / nonzero_user_count,
        "bias"      : -1,
        "concentration" : -1
    }

def stratified(infilename, infilename_neg, trainfilename, propensities, K=30, partition=10, delta=0.1):

    # Read pickles
    infile = open(infilename, 'rb')
    infile_neg = open(infilename_neg, 'rb')
    P = pickle.load(infile)
    infile.close()
    P_neg = pickle.load(infile_neg)
    infile_neg.close()
    NUM_NEGATIVES = P["num_negatives"]

    # Merge P and P_neg
    for theuser in P["users"]:
        neg_items = list(P_neg["user_items"][theuser][NUM_NEGATIVES:])
        neg_scores = list(P_neg["results"][theuser][NUM_NEGATIVES:])
        P["user_items"][theuser] = list(neg_items) + list(P["user_items"][theuser][NUM_NEGATIVES:])
        P["results"][theuser] = list(neg_scores) + list(P["results"][theuser][NUM_NEGATIVES:])

    Zui = dict()
    Ni = dict()

    # Compute frequencies of items in the training set
    trainset = np.load(trainfilename)
    for i in trainset['item_id']:
        if i in Ni:
            Ni[i] += 1
        else:
            Ni[i] = 1

    # Compute recommendations for each user
    for theuser in P["users"]:
        all_scores = np.array(P["results"][theuser])
        pos_items = P["user_items"][theuser][len(P_neg["user_items"][theuser][NUM_NEGATIVES:]):]
        pos_scores = P["results"][theuser][len(P_neg["results"][theuser][NUM_NEGATIVES:]):]
        for i, pos_item in enumerate(pos_items):
            pos_score = pos_scores[i]
            Zui[(theuser, pos_item)] = float(np.sum(all_scores > pos_score))

    w = dict()

    # Store the ids of the items to be used to compute the bias
    items_of_the_test_set = set()
    for theuser in P["users"]:
        for theitem in P["user_items"][theuser][len(P_neg["user_items"][theuser][NUM_NEGATIVES:]):]:
            if theitem not in items_of_the_test_set:
                items_of_the_test_set.add(theitem)

   
    # Using as pui a single row of propensities, as we assumed propensities to be user independent
    pui = propensities[0,:]

    # Take the list of items (not tuples) in pui sorted by value
    items_sorted_by_value = np.argsort(pui)[::-1]

    # Remove items not of the testset
    items_sorted_by_value = np.array([item for item in items_sorted_by_value if item in items_of_the_test_set])

    # Filter out indices of the items not in the test set
    items_sorted_by_value = items_sorted_by_value[pui[items_sorted_by_value] > 0]

    # Compute linspace between the pui[0] and pui[-1] 
    linspace = np.linspace(pui[items_sorted_by_value[0]], pui[items_sorted_by_value[-1]], partition+1)
   
    # Compute dictionary w, that is, for each item, assigns the average of the puis in the partition it belongs to
    i = 0
    j = 0
    while i < len(items_sorted_by_value):
        # Init average, start and end
        avg = 0
        start = i
        end = i
    
        # Compute average pui for the subset
        while i < len(items_sorted_by_value) and pui[items_sorted_by_value[i]] >= linspace[j+1]:
            avg += 1.0 / pui[items_sorted_by_value[i]]
            end = i
            i += 1
        avg = avg / (end - start + 1)

        # Assign the average to the items in the subset
        for k in range(start, end+1):
            w[items_sorted_by_value[k]] = avg

        # Move to the next subset of the partition
        j += 1

    # Compute bias' numerator
    bias = 0.0
    for k in items_sorted_by_value:
        # add |pui*w - 1!|
        bias += abs(pui[k] * w[k] - 1)
    # Multiply by number of users
    bias *= len(P["users"])

    # Compute concentrations numerator (for each user)
    concentrations = {}
    max_w = max(w.values())
    # ... by computing the sum of squares of w for each user
    for user, item in zip(trainset['user_id'], trainset['item_id']):
        # Iterate over the trainset to compute the sum of squares for each user
        if item in w:
            if user not in concentrations:
                concentrations[user] = 0
            concentrations[user] += w[item] ** 2
    # ... and then applying the formula
    for user in concentrations:
        concentrations[user] = math.sqrt(concentrations[user] * 2 * math.log(2/delta)) + max_w * 7 * math.log(2/delta)
    # Now sum all the concentrations
    concentration = sum(concentrations.values())

    # Delete trainset
    del trainset

    # Calculate per-user scores
    nonzero_user_count = 0
    sum_user_auc = 0.0
    sum_user_recall = 0.0
    for theuser in P["users"]:
        numerator_auc = 0.0
        numerator_recall = 0.0
        denominator = 0.0
        
        for theitem in P["user_items"][theuser][len(P_neg["user_items"][theuser][NUM_NEGATIVES:]):]:
            # Skip items with null frequency
            if  theitem not in Ni:
                continue

            # Add things to be summed for each item
            numerator_auc += (1 - Zui[(theuser, theitem)] / len(P["user_items"][theuser])) * w[theitem]

            # Add things for recall
            if Zui[(theuser, theitem)] < K:
                numerator_recall += 1.0 * w[theitem]

            # Increment denominator that the sum must be divided by 
            denominator += 1 / pui[theitem]

        # If there was at least one item for the user, count the user and sum the results
        if denominator > 0:
            nonzero_user_count += 1
            sum_user_auc += numerator_auc / denominator
            sum_user_recall += numerator_recall / denominator 

    # Return
    return {
        "auc"       : sum_user_auc / nonzero_user_count, 
        "recall"    : sum_user_recall / nonzero_user_count,
        "bias"      : bias,
        "concentration" : concentration
    }

def stratified_logspace(infilename, infilename_neg, trainfilename, propensities, K=30, partition=10, delta=0.1):

    # Read pickles
    infile = open(infilename, 'rb')
    infile_neg = open(infilename_neg, 'rb')
    P = pickle.load(infile)
    infile.close()
    P_neg = pickle.load(infile_neg)
    infile_neg.close()
    NUM_NEGATIVES = P["num_negatives"]

    # Merge P and P_neg
    for theuser in P["users"]:
        neg_items = list(P_neg["user_items"][theuser][NUM_NEGATIVES:])
        neg_scores = list(P_neg["results"][theuser][NUM_NEGATIVES:])
        P["user_items"][theuser] = list(neg_items) + list(P["user_items"][theuser][NUM_NEGATIVES:])
        P["results"][theuser] = list(neg_scores) + list(P["results"][theuser][NUM_NEGATIVES:])

    Zui = dict()
    Ni = dict()

    # Compute frequencies of items in the training set
    trainset = np.load(trainfilename)
    for i in trainset['item_id']:
        if i in Ni:
            Ni[i] += 1
        else:
            Ni[i] = 1

    # Compute recommendations for each user
    for theuser in P["users"]:
        all_scores = np.array(P["results"][theuser])
        pos_items = P["user_items"][theuser][len(P_neg["user_items"][theuser][NUM_NEGATIVES:]):]
        pos_scores = P["results"][theuser][len(P_neg["results"][theuser][NUM_NEGATIVES:]):]
        for i, pos_item in enumerate(pos_items):
            pos_score = pos_scores[i]
            Zui[(theuser, pos_item)] = float(np.sum(all_scores > pos_score))

    w = dict()

    # Store the ids of the items to be used to compute the bias
    items_of_the_test_set = set()
    for theuser in P["users"]:
        for theitem in P["user_items"][theuser][len(P_neg["user_items"][theuser][NUM_NEGATIVES:]):]:
            if theitem not in items_of_the_test_set:
                items_of_the_test_set.add(theitem)

   
    # Using as pui a single row of propensities, as we assumed propensities to be user independent
    pui = propensities[0,:]

    # Take the list of items (not tuples) in pui sorted by value
    items_sorted_by_value = np.argsort(pui)[::-1]

    # Remove items not of the testset
    items_sorted_by_value = np.array([item for item in items_sorted_by_value if item in items_of_the_test_set])

    # Filter out indices of the items not in the test set
    items_sorted_by_value = items_sorted_by_value[pui[items_sorted_by_value] > 0]

    # Compute logspace between the pui[0] and pui[-1] 
    logspace = np.logspace(np.log10(pui[items_sorted_by_value[0]]), np.log10(pui[items_sorted_by_value[-1]]), partition+1)
   
    # Compute dictionary w, that is, for each item, assigns the average of the puis in the partition it belongs to
    i = 0
    j = 0
    while i < len(items_sorted_by_value):
        # Init average, start and end
        avg = 0
        start = i
        end = i

         # Ensure that j does not exceed logspace size
        next_boundary = logspace[j+1] if j + 1 < len(logspace) else float('inf')
    
        # Compute average pui for the subset
        while i < len(items_sorted_by_value) and pui[items_sorted_by_value[i]] >= next_boundary:
            avg += 1.0 / pui[items_sorted_by_value[i]]
            end = i
            i += 1


        avg = avg / (end - start + 1)

        # Assign the average to the items in the subset
        for k in range(start, end+1):
            w[items_sorted_by_value[k]] = avg

        # Move to the next subset of the partition
        j += 1

    # Compute bias' numerator
    bias = 0.0
    for k in items_sorted_by_value:
        # add |pui*w - 1!|
        bias += abs(pui[k] * w[k] - 1)
    # Multiply by number of users
    bias *= len(P["users"])

    # Compute concentrations numerator (for each user)
    concentrations = {}
    max_w = max(w.values())
    # ... by computing the sum of squares of w for each user
    for user, item in zip(trainset['user_id'], trainset['item_id']):
        # Iterate over the trainset to compute the sum of squares for each user
        if item in w:
            if user not in concentrations:
                concentrations[user] = 0
            concentrations[user] += w[item] ** 2
    # ... and then applying the formula
    for user in concentrations:
        concentrations[user] = math.sqrt(concentrations[user] * 2 * math.log(2/delta)) + max_w * 7 * math.log(2/delta)
    # Now sum all the concentrations
    concentration = sum(concentrations.values())

    # Delete trainset
    del trainset

    # Calculate per-user scores
    nonzero_user_count = 0
    sum_user_auc = 0.0
    sum_user_recall = 0.0
    for theuser in P["users"]:
        numerator_auc = 0.0
        numerator_recall = 0.0
        denominator = 0.0
        
        for theitem in P["user_items"][theuser][len(P_neg["user_items"][theuser][NUM_NEGATIVES:]):]:
            # Skip items with null frequency
            if  theitem not in Ni:
                continue

            # Add things to be summed for each item
            numerator_auc += (1 - Zui[(theuser, theitem)] / len(P["user_items"][theuser])) * w[theitem]

            # Add things for recall
            if Zui[(theuser, theitem)] < K:
                numerator_recall += 1.0 * w[theitem]

            # Increment denominator that the sum must be divided by 
            denominator += 1 / pui[theitem]

        # If there was at least one item for the user, count the user and sum the results
        if denominator > 0:
            nonzero_user_count += 1
            sum_user_auc += numerator_auc / denominator
            sum_user_recall += numerator_recall / denominator 

    # Return
    return {
        "auc"       : sum_user_auc / nonzero_user_count, 
        "recall"    : sum_user_recall / nonzero_user_count,
        "bias"      : bias,
        "concentration" : concentration
    }


    

# This version uses the linspace of the number of number of items used for evaluation, not of the propensities
def stratified_2(infilename, infilename_neg, trainfilename, propensities, K=30, partition=10, delta=0.1):

    # Read pickles
    infile = open(infilename, 'rb')
    infile_neg = open(infilename_neg, 'rb')
    P = pickle.load(infile)
    infile.close()
    P_neg = pickle.load(infile_neg)
    infile_neg.close()
    NUM_NEGATIVES = P["num_negatives"]

    # Merge P and P_neg
    for theuser in P["users"]:
        neg_items = list(P_neg["user_items"][theuser][NUM_NEGATIVES:])
        neg_scores = list(P_neg["results"][theuser][NUM_NEGATIVES:])
        P["user_items"][theuser] = list(neg_items) + list(P["user_items"][theuser][NUM_NEGATIVES:])
        P["results"][theuser] = list(neg_scores) + list(P["results"][theuser][NUM_NEGATIVES:])

    Zui = dict()
    Ni = dict()

    # Compute frequencies of items in the training set
    trainset = np.load(trainfilename)
    for i in trainset['item_id']:
        if i in Ni:
            Ni[i] += 1
        else:
            Ni[i] = 1

    # Compute recommendations for each user
    for theuser in P["users"]:
        all_scores = np.array(P["results"][theuser])
        pos_items = P["user_items"][theuser][len(P_neg["user_items"][theuser][NUM_NEGATIVES:]):]
        pos_scores = P["results"][theuser][len(P_neg["results"][theuser][NUM_NEGATIVES:]):]
        for i, pos_item in enumerate(pos_items):
            pos_score = pos_scores[i]
            Zui[(theuser, pos_item)] = float(np.sum(all_scores > pos_score))

    w = dict()

    # Store the ids of the items to be used to compute the bias
    items_of_the_test_set = set()
    for theuser in P["users"]:
        for theitem in P["user_items"][theuser][len(P_neg["user_items"][theuser][NUM_NEGATIVES:]):]:
            if theitem not in items_of_the_test_set:
                items_of_the_test_set.add(theitem)

   
    # Using as pui a single row of propensities, as we assumed propensities to be user independent
    pui = propensities[0,:]

    # Take the list of items (not tuples) in pui sorted by value
    items_sorted_by_value = np.argsort(pui)[::-1]

    # Remove items not of the testset
    items_sorted_by_value = np.array([item for item in items_sorted_by_value if item in items_of_the_test_set])

    # Filter out indices of the items not in the test set
    items_sorted_by_value = items_sorted_by_value[pui[items_sorted_by_value] > 0]

    # Compute linspace between the 0 to len(item_sorted...)
    linspace = np.linspace(0, len(items_sorted_by_value), partition+1)
   
    # Compute dictionary w, that is, for each item, assigns the average of the puis in the partition it belongs to
    i  =0
    j = 0
    while i < len(items_sorted_by_value):
        # Init average, start and end                    
        avg = 0
        start = i
        end = i
    
        # Compute average pui for the subset
        while i < len(items_sorted_by_value) and i < linspace[j+1]:
            avg += 1.0 / pui[items_sorted_by_value[i]]
            end = i
            i += 1
        avg = avg / (end - start + 1)

        # Assign the average to the items in the subset
        for k in range(start, end+1):
            w[items_sorted_by_value[k]] = avg

        # Move to the next subset of the partition
        j += 1

    # Compute bias' numerator
    bias = 0.0
    for k in items_sorted_by_value:
        # add |pui*w - 1!|
        bias += abs(pui[k] * w[k] - 1)
    # Multiply by number of users
    bias *= len(P["users"])

    # Compute concentrations numerator (for each user)
    concentrations = {}
    max_w = max(w.values())
    # ... by computing the sum of squares of w for each user
    for user, item in zip(trainset['user_id'], trainset['item_id']):
        # Iterate over the trainset to compute the sum of squares for each user
        if item in w:
            if user not in concentrations:
                concentrations[user] = 0
            concentrations[user] += w[item] ** 2
    # ... and then applying the formula
    for user in concentrations:
        concentrations[user] = math.sqrt(concentrations[user] * 2 * math.log(2/delta)) + max_w * 7 * math.log(2/delta)
    # Now sum all the concentrations
    concentration = sum(concentrations.values())

    # Calculate per-user scores
    nonzero_user_count = 0
    sum_user_auc = 0.0
    sum_user_recall = 0.0
    for theuser in P["users"]:
        numerator_auc = 0.0
        numerator_recall = 0.0
        denominator = 0.0
        
        for theitem in P["user_items"][theuser][len(P_neg["user_items"][theuser][NUM_NEGATIVES:]):]:
            # Skip items with null frequency
            if  theitem not in Ni:
                continue

            # Add things to be summed for each item
            numerator_auc += (1 - Zui[(theuser, theitem)] / len(P["user_items"][theuser])) * w[theitem]

            # Add things for recall
            if Zui[(theuser, theitem)] < K:
                numerator_recall += 1.0 * w[theitem]

            # Increment denominator that the sum must be divided by 
            denominator += 1 / pui[theitem]

        # If there was at least one item for the user, count the user and sum the results
        if denominator > 0:
            nonzero_user_count += 1
            sum_user_auc += numerator_auc / denominator
            sum_user_recall += numerator_recall / denominator 

    # Return
    return {
        "auc"       : sum_user_auc / nonzero_user_count, 
        "recall"    : sum_user_recall / nonzero_user_count,
        "bias"      : bias,
        "concentration" : concentration
    }
