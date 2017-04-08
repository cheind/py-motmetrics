         
import clearmot as cm

if __name__== '__main__':
    
    df_gt = cm.io.loadtxt('etc/data/TUD-Campus/gt.txt')
    df_test = cm.io.loadtxt('etc/data/TUD-Campus/test.txt')

    acc = cm.new_accumulator()

    for frameid, dff_gt in df_gt.groupby(level=0):
        dff_gt = dff_gt.loc[frameid]
        if frameid in df_test.index:
            dff_test = df_test.loc[frameid]

            hids = dff_test.index.values
            oids = dff_gt.index.values

            hrects = dff_test[['x', 'y', 'w', 'h']].values
            orects = dff_gt[['x', 'y', 'w', 'h']].values

            dists = cm.distances.iou_matrix(orects, hrects, max_iou=0.5)

            cm.update_mot(acc, oids, hids, dists, frameid=frameid)

    cm.print_stats(acc)