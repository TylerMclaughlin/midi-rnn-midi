# midi-rnn-midi

This project uses deep learning to capture the relationship between musical phrases played by human composers.  

Single channel, linear rhythms mode uses the python-midi package.  This is pretty much just for drums.

Extending to multi-channel compositions, I made use of the pretty midi package.

## Data Augmentation Strategies

Sparsifying data is useful if you have a small training set with complex musical rhythms or passages.

Transposing notes that are not drums.

Injecting digitized noise into midi files, affecting the note timing with a Poisson process, but such that the notes are pushed backward after everytime they are pushed forward a tick.

