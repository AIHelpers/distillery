package memory

import "distillery/internal/domain"

type FeedbackRepo struct{ store *Store }

func NewFeedbackRepo(store *Store) *FeedbackRepo { return &FeedbackRepo{store: store} }

func (r *FeedbackRepo) Add(f *domain.Misprediction) error {
	r.store.mu.Lock()
	defer r.store.mu.Unlock()
	r.store.Feedback[f.TaskID] = append(r.store.Feedback[f.TaskID], f)
	r.store.persist()
	return nil
}

func (r *FeedbackRepo) ListByTask(taskID string) ([]*domain.Misprediction, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()
	list := r.store.Feedback[taskID]
	out := make([]*domain.Misprediction, len(list))
	copy(out, list)
	return out, nil
}

func (r *FeedbackRepo) ListUnresolved(taskID string) ([]*domain.Misprediction, error) {
	r.store.mu.RLock()
	defer r.store.mu.RUnlock()
	var out []*domain.Misprediction
	for _, f := range r.store.Feedback[taskID] {
		if !f.Resolved {
			out = append(out, f)
		}
	}
	return out, nil
}

func (r *FeedbackRepo) MarkResolved(ids []string) error {
	if len(ids) == 0 {
		return nil
	}
	set := make(map[string]bool, len(ids))
	for _, id := range ids {
		set[id] = true
	}
	r.store.mu.Lock()
	defer r.store.mu.Unlock()
	for _, list := range r.store.Feedback {
		for _, f := range list {
			if set[f.ID] {
				f.Resolved = true
			}
		}
	}
	r.store.persist()
	return nil
}
