def clusters_selection_parametrs(data, X_tag, y_tag, cluster_alg = 'Kmeans'):
    
    if cluster_alg == 'Kmeans':    
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        import numpy as np

        X = data[X_tag] 
        if y_tag != None:
            y = data[y_tag]
        
        s_score_list = []
        v_score_list = []
        
        eps_list = np.linspace(0.1, 1, 10)
        min_samples_list = [2, 3,  5, 8, 13]
        result = []
        
        
        for sample in min_samples_list:
            if  y_tag != None:
                clustering =KMeans(n_clusters = sample).fit(X)
            else:
                clustering = KMeans(n_clusters = sample).fit(X)
            
            s_score = silhouette_score(X, clustering.labels_)
            result_ = ["метрика силуэтта %.3f" % s_score,
                        
                        "min_samples V %.0f" % sample, ]
            result.append(result_)

    else:
        
        from sklearn.cluster import DBSCAN
        from sklearn.metrics import silhouette_score
        from sklearn.metrics.cluster import v_measure_score
        X = data[X_tag] 
        if y_tag != None:
            y = data[y_tag]
        
        s_score_list = []
        v_score_list = []
        
        eps_list = np.linspace(0.1, 1, 10)
        min_samples_list = [2, 3,  5, 8, 13]
        result = []
        
        
        for sample in min_samples_list:
                
            for eps in eps_list:
                if  y_tag != None:
                    clustering = DBSCAN(eps=eps, min_samples=sample).fit(X, y)
                else:
                    clustering = DBSCAN(eps=eps, min_samples=sample).fit(X)
                s_score = silhouette_score(X, clustering.labels_)
                #v_score = v_measure_score(y, clustering.labels_)
                
                result_ = ["метрика силуэтта %.3f" % s_score,
                        #"метрика V %.3f" % v_score, 
                        "eps  %.1f" % eps,
                        "min_samples V %.0f" % sample, ]
                result.append(result_)
        
           
    return (result)