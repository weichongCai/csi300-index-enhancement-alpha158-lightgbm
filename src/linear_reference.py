"""A fixed Ridge benchmark; train-only preprocessing, streamed normal equations."""
import numpy as np


def blocks(frame, positions, size=8192):
    for start in range(0, len(positions), size):
        yield start, frame.iloc[positions[start:start+size]].to_numpy(dtype=float)


def fit_ridge(frame, labels, positions, penalty=1., clip=5.):
    labels=np.asarray(labels,dtype=float)
    if len(labels)!=len(positions) or not np.isfinite(labels).all():
        raise ValueError('Invalid training targets.')
    width=frame.shape[1]
    count=np.zeros(width);total=np.zeros(width)
    for _,x in blocks(frame,positions):
        valid=np.isfinite(x);count+=valid.sum(axis=0);total+=np.where(valid,x,0).sum(axis=0)
    mean=np.divide(total,count,out=np.zeros(width),where=count>0)
    ss=np.zeros(width)
    for _,x in blocks(frame,positions):
        ss+=np.where(np.isfinite(x),(x-mean)**2,0).sum(axis=0)
    scale=np.sqrt(np.divide(ss,count,out=np.zeros(width),where=count>0))
    scale[(scale<1e-12)|~np.isfinite(scale)]=1.
    gram=np.zeros((width+1,width+1));rhs=np.zeros(width+1)
    for start,x in blocks(frame,positions):
        z=transform(x,mean,scale,clip)
        a=np.column_stack([z,np.ones(len(z))])
        gram+=a.T@a;rhs+=a.T@labels[start:start+len(a)]
    regularizer=np.eye(width+1)*penalty;regularizer[-1,-1]=0.
    coefficient=np.linalg.solve(gram/len(labels)+regularizer,rhs/len(labels))
    return {'feature_names':list(frame.columns),'mean':mean.tolist(),'scale':scale.tolist(),
            'coefficient':coefficient[:-1].tolist(),'intercept':float(coefficient[-1]),
            'clip':clip,'lambda_average_mse':penalty,'training_rows':len(labels),
            'objective':'mean((target-intercept-z@coefficient)^2) + lambda*sum(coefficient^2)',
            'imputation':'training feature mean, then standardized and clipped',
            'preprocessing_fit_on_training_only':True}


def transform(values,mean,scale,clip):
    values=np.asarray(values,dtype=float)
    z=(np.where(np.isfinite(values),values,mean)-mean)/scale
    return np.clip(z,-clip,clip)


def predict_ridge(frame,model,positions=None):
    if list(frame.columns)!=model['feature_names']:
        raise ValueError('Feature names/order changed.')
    if positions is None:positions=np.arange(len(frame))
    mean=np.asarray(model['mean']);scale=np.asarray(model['scale']);coef=np.asarray(model['coefficient'])
    result=np.empty(len(positions))
    for start,x in blocks(frame,positions):
        result[start:start+len(x)]=transform(x,mean,scale,model['clip'])@coef+model['intercept']
    return result
