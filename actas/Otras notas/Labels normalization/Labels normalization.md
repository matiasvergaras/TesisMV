# Labels normalization

What we have so far 

- Normalization as "the process of creating equivalence classes of different types".

- We normalized the labels to get their *term* via **lemmatization**
- We also applied **Stopword Removal** to reduce the size of the vocabulary and eliminate terms that do not provide information
- We used NLTK library.
- Zipf's Law: the frequency *f* of a term in a corpus is inversely proportional to its ranking *r* in a sorted frequency table.



Testing the new label set against the same experiments, we got:

- A label cardinality of 12 (vs 4)
- A label density of 0.015 (vs 0.005)
- The number of labels with a frequency greater than 15 went from 26 to 53
- Since there are now more labels for each threshold, the exact match ratio dropped considerably.
  - ...But Hamming Score remains around 0.5 at *t=15*. However, its growth is now much slower.
  - By looking at the confusion matrices, more useful or plausible results begin to emerge. Algorithms fail more, but the problem is becoming more real.

What should we focus on in the next few weeks:

- Data augmentation? 
- Class imbalance. How to treat the imbalance in favor of the most present labels without losing those that accompany them?


---


Yes, we have some results to comment on.

Regarding the label prediction, these days we have been working on its normalization. For this we decided to apply lemmatization through WordNet, which reduced the number of labels from 586 to 339 (57.85%). This is good, is what we wanted, to have less labels. However to achieve that we took some decisions, like to bring together the plural and singular versions of a same label in a single singular option, and also separate the compound labels (like vertical panel) into simple ones (in that case, vertical and panel per separate).

As a result, the label density (one of our main concerns in the previous presentation) increased from 0.005 to 0.015 (three times the previous value), the label cardinality went also from 4 to 12, and the number of labels with at least 15 occurrences increased from 26 to 53 (so now we are able to work with the double of labels).

However, the predictions are still not good enough, and we believe that it may be due to the large imbalance of negative and positive cases that still exists for each label. The next few days we will be working on this aspect to see if we finally get good classifications. 

The problem remains interesting, we have been researching and applying different techniques and we believe that we still have a lot to test and to improve.

Just that, mostly.