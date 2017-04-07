         
import clearmot as cm

if __name__== '__main__':
    
    acc = cm.new_mot(auto_id=True)
    
    cm.update_mot(acc, [1, 2, 3, 4], [], [])
    cm.update_mot(acc, [1, 2, 3, 4], [], [])
    cm.update_mot(acc, [1, 2, 3, 4], [], [])
    cm.update_mot(acc, [1, 2, 3, 4], [], [])

    cm.update_mot(acc, [4], [4], [[0]])
    cm.update_mot(acc, [4], [4], [[0]])
    cm.update_mot(acc, [4], [4], [[0]])
    cm.update_mot(acc, [4], [4], [[0]])

    cm.stats.print_stats([acc.events, acc.events.loc[1:3]], names=['a', 'b'])