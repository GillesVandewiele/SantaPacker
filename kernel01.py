coal=0
book=0
bike=0
train=0
blocks=0
doll=0
horse=0
gloves=0
ball=0
with open("Santa_03.csv", 'w') as f:
        f.write("Gifts\n")
        for i in range(1000):
            if coal < 166:
                f.write('coal_'+str(coal)+' book_'+str(book))
                coal+=1
                book+=1
                f.write(' book_'+str(book)+'\n')
                book+=1
            elif blocks < 1000 and train < 1000:
                f.write('blocks_'+str(blocks)+' train_'+str(train))
                blocks+=1
                train+=1
                f.write(' blocks_'+str(blocks)+' train_'+str(train)+'\n')
                blocks+=1
                train+=1
            elif bike < 500 and blocks < 1000:
                f.write('bike_'+str(bike)+' train_'+str(train)+' blocks_'+str(blocks)+'\n')
                bike+=1
                train+=1
                blocks+=1
            elif book < 1000 and gloves < 200:
                f.write('doll_'+str(doll))
                doll+=1
                f.write(' doll_'+str(doll))
                doll+=1
                f.write(' doll_'+str(doll))
                doll+=1
                f.write(' horse_'+str(horse))
                horse+=1
                f.write(' horse_'+str(horse))
                horse+=1
                f.write(' horse_'+str(horse))
                horse+=1
                f.write(' gloves_'+str(gloves))
                gloves+=1
                f.write(' ball_'+str(ball))
                ball+=1
                f.write(' ball_'+str(ball))
                ball+=1
                f.write(' ball_'+str(ball))
                ball+=1
                f.write(' ball_'+str(ball))
                ball+=1
                f.write(' ball_'+str(ball))
                ball+=1
                f.write(' book_'+str(book)+'\n')
                book+=1
            # (1) bike  (4) books  (2) horse -- See Graph below
            elif bike < 500 and horse < 1000 and book < 1200:
                f.write('bike_'+str(bike))
                bike+=1
                f.write(' book_'+str(book))
                book+=1
                f.write(' book_'+str(book))
                book+=1
                f.write(' book_'+str(book))
                book+=1
                f.write(' book_'+str(book))
                book+=1
                f.write(' horse_'+str(horse))
                horse+=1
                f.write(' horse_'+str(horse)+'\n')
                horse+=1

with open("Santa_BTB.csv", 'w') as f:
    f.write("Gifts\n")
    for i in range(1000):
        f.write('train_' + str(i) + ' blocks_' + str(i) + ' horse_' + str(i) + ' doll_' + str(i) + ' book_' + str(
            i) + ' ball_' + str(i) + '\n')